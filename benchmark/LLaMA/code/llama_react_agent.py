import re
import string
import os
import time
from typing import List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub.hf_api import HfFolder
from langchain_core.language_models.llms import BaseLLM
from langchain_core.prompts import PromptTemplate
import tiktoken

# Import from ReAct code
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../ReAct/code'))
from prompts import react_agent_prompt
from fewshots import TOOLQA_EASY8, TOOLQA_HARD3
from tools.math import calculator
# Lazy imports for agenda_retriever and scirex_retriever (they require chromadb which needs newer sqlite3)
# These are only imported when actually needed (for agenda/scirex datasets)
from tools.table import tabtools
from tools.graph import graphtools
from tools.code import sql_interpreter, python_interpreter


def parse_action(string):
    pattern = r'^(\w+)\[(.+)\]$'
    match = re.match(pattern, string)
    
    if match:
        action_type = match.group(1)
        argument = match.group(2)
        return action_type, argument
    
    else:
        return None, None


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the|USD)\b", " ", text)
    
    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def EM(answer, key) -> bool:
    return normalize_answer(answer) == normalize_answer(key)


def format_step(step: str) -> str:
    return step.strip('\n').strip().replace('\n', '')


class LlamaLLM(BaseLLM):
    # Configure Pydantic to allow extra fields
    class Config:
        extra = "allow"
    
    def __init__(self, pipe, tokenizer, max_tokens=100, temperature=0.0):
        super().__init__()
        # Use object.__setattr__ to bypass Pydantic validation for custom attributes
        object.__setattr__(self, 'pipe', pipe)
        object.__setattr__(self, 'tokenizer', tokenizer)
        object.__setattr__(self, 'max_tokens', max_tokens)
        object.__setattr__(self, 'temperature', temperature)
        object.__setattr__(self, 'model', pipe.model)  # Store model for direct access
    
    def _call_with_logprobs(self, prompt: str, stop: Optional[List[str]] = None) -> tuple:
        """Call model and return both text and per-token log probabilities"""
        from transformers import StoppingCriteria, StoppingCriteriaList
        
        # Don't use chat template - the prompt already has the proper format
        # The ReAct prompt is designed to work without chat formatting
        # Just tokenize the prompt directly
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        
        # Create stopping criteria for stop sequences
        class StopOnTokens(StoppingCriteria):
            def __init__(self, tokenizer, stop_sequences):
                self.tokenizer = tokenizer
                self.stop_sequences = stop_sequences
                self.stop_ids = set()
                # Get newline token ID - try both with and without special tokens
                newline_token_id = tokenizer.encode("\n", add_special_tokens=False)
                if len(newline_token_id) > 0:
                    self.stop_ids.add(newline_token_id[0])
                # Also try EOS token
                if tokenizer.eos_token_id is not None:
                    self.stop_ids.add(tokenizer.eos_token_id)
                # Also check for other stop sequences
                for stop_seq in stop_sequences:
                    if stop_seq == "\n":
                        continue  # Already handled
                    stop_ids = tokenizer.encode(stop_seq, add_special_tokens=False)
                    for stop_id in stop_ids:
                        self.stop_ids.add(stop_id)
                
            def __call__(self, input_ids, scores, **kwargs):
                # Only check stop sequences if we've generated at least 1 token
                # This prevents stopping immediately on the first token
                if len(input_ids[0]) > 0:
                    # Count generated tokens (approximate - input length would need to be tracked)
                    # For now, just check if we have enough tokens
                    last_token = input_ids[0][-1].item()
                    if last_token in self.stop_ids:
                        # Only stop if this isn't the very first generated token
                        # We'll check the generated length in post-processing instead
                        return True
                return False
        
        stopping_criteria = None
        if stop:
            stopping_criteria = StoppingCriteriaList([StopOnTokens(self.tokenizer, stop)])
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=min(self.max_tokens, 100),  # Increased to allow complete actions
                temperature=self.temperature if self.temperature > 0 else 1.0,
                do_sample=self.temperature > 0,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=stopping_criteria
            )
        
        # Get generated tokens and compute log probs
        generated_ids = outputs.sequences[0][input_ids.shape[1]:]
        logprobs = []
        
        for i, token_id in enumerate(generated_ids):
            if i < len(outputs.scores):
                logits = outputs.scores[i][0]
                log_probs = torch.log_softmax(logits, dim=-1)
                logprobs.append(log_probs[token_id].item())
        
        # Decode only the generated part (not the full sequence)
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_text = generated_text.strip()
        
        import re
        
        # Handle stop sequences (post-process as backup) - do this more aggressively
        # Split on newline and take only the first line (before any newline)
        if stop and "\n" in stop:
            lines = generated_text.split("\n")
            generated_text = lines[0] if lines else generated_text
            generated_text = generated_text.strip()
        
        # Also handle other stop sequences
        if stop:
            for stop_seq in stop:
                if stop_seq != "\n" and stop_seq in generated_text:
                    # Split and take only the first part
                    parts = generated_text.split(stop_seq)
                    generated_text = parts[0]
                    break  # Stop at first match
        
        # Clean up any instruction text that might have been generated
        # The model sometimes repeats instruction text - remove it
        instruction_phrases = [
            "To answer the question",
            "we need to follow",
            "interleaving Thought, Action, Observation",
            "Solve a question answering task"
        ]
        for phrase in instruction_phrases:
            if phrase in generated_text:
                # If the entire text is just instruction repetition, return empty
                if generated_text.strip().startswith(phrase) or generated_text.strip() == phrase:
                    return "", logprobs
                # Otherwise try to extract just the actual content after the phrase
                parts = generated_text.split(phrase)
                if len(parts) > 1:
                    generated_text = parts[-1].strip()
        
        # Clean up format labels that the model might generate (e.g., "Thought 1:", "Action 1:")
        import re
        # Remove patterns like "Thought 1:", "Action 2:", etc. - be more aggressive
        generated_text = re.sub(r'^(Thought|Action)\s+\d+\s*:\s*', '', generated_text, flags=re.IGNORECASE)
        generated_text = re.sub(r'\n(Thought|Action)\s+\d+\s*:\s*', ' ', generated_text, flags=re.IGNORECASE)
        # Also remove if it appears anywhere in the text
        generated_text = re.sub(r'(Thought|Action)\s+\d+\s*:\s*', '', generated_text, flags=re.IGNORECASE)
        
        # If the text is just format labels or very short instruction text, return empty
        if len(generated_text.strip()) < 5:
            return "", logprobs
        
        # If the generated text is still instruction-like and long, truncate it
        # Don't return empty as that would break the agent - instead try to extract useful part
        if any(phrase in generated_text.lower() for phrase in ["to answer", "we need to", "interleaving"]):
            if len(generated_text) > 50:  # Long instruction text
                # Try to find the last sentence or meaningful part
                sentences = generated_text.split('.')
                if len(sentences) > 1:
                    # Take the last sentence that's not instruction-like
                    for sent in reversed(sentences):
                        sent = sent.strip()
                        if sent and not any(p in sent.lower() for p in ["to answer", "we need to", "interleaving"]):
                            generated_text = sent
                            break
                    else:
                        # If all sentences are instruction-like, just take a short part
                        generated_text = generated_text[:50]
        
        return generated_text.strip(), logprobs
    
    def sample_answers(self, prompt: str, n_samples: int = 10, 
                      temperature: float = 0.8, stop: Optional[List[str]] = None) -> List[tuple]:
        """Sample N answers with their log probabilities"""
        samples = []
        original_temp = self.temperature
        object.__setattr__(self, 'temperature', temperature)
        
        for _ in range(n_samples):
            text, logprobs = self._call_with_logprobs(prompt, stop)
            samples.append((text, logprobs))
        
        object.__setattr__(self, 'temperature', original_temp)
        return samples
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call the model with the given prompt"""
        text, _ = self._call_with_logprobs(prompt, stop)
        return text
    
    def __call__(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Make the LLM callable (required by langchain BaseLLM interface)"""
        return self._call(prompt, stop)
    
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager = None,
        **kwargs,
    ):
        """Generate responses for a list of prompts (required by BaseLLM)"""
        from langchain_core.outputs import LLMResult, Generation
        
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop)
            generations.append([Generation(text=text)])
        
        return LLMResult(generations=generations)
    
    @property
    def _llm_type(self) -> str:
        return "llama"


class ReactAgent:
    def __init__(self,
                 args,
                 question: str,
                 key: str,
                 max_steps: int = 20,
                 agent_prompt: PromptTemplate = react_agent_prompt,
                 llama_model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
                 hf_token: Optional[str] = None,
                 preloaded_pipe = None,
                 preloaded_tokenizer = None,
                 ) -> None:
        
        self.question = question
        self.answer = ''
        self.key = key
        self.max_steps = max_steps
        self.agent_prompt = agent_prompt
        if args.prompt == "easy":
            self.react_examples = TOOLQA_EASY8
        else:
            self.react_examples = TOOLQA_HARD3
        
        # Add entropy tracking (after react_examples is set)
        # Count few-shot examples
        fewshot_count = len(self.react_examples.split('Question:')) - 1 if self.react_examples else 0
        
        self.entropy_data = {
            'tool_calls': [],
            'tool_entropies': [],
            'answer_samples': [],
            'answer_logprobs': [],
            'predictive_entropy': None,
            'semantic_entropy': None,
            'sta_predictive': None,
            'sta_semantic': None,
            # Step-wise entropy for few-shot analysis
            'step_entropies': [],  # List of dicts: {step, type: 'thought'|'action', entropy, samples, logprobs}
            'fewshot_examples': fewshot_count
        }

        # Setup Llama model - reuse if provided, otherwise load new
        if preloaded_pipe is not None and preloaded_tokenizer is not None:
            # Reuse preloaded model
            pipe = preloaded_pipe
            tokenizer = preloaded_tokenizer
        else:
            # Load new model (for debug mode or single question)
            if hf_token:
                HfFolder.save_token(hf_token)
            
            dtype = torch.bfloat16
            tokenizer = AutoTokenizer.from_pretrained(llama_model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                llama_model_name,
                dtype=dtype,
                device_map="auto",
                max_memory={0: "20GiB"},
                trust_remote_code=True
            )

            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto",
                dtype=dtype,
                model_kwargs={"temperature": 0.0, "do_sample": False}
            )
        
        # Initialize LLM with stop sequence configured (like ReAct does)
        # The stop sequence "\n" is handled in _call_with_logprobs
        self.llm = LlamaLLM(pipe, tokenizer, max_tokens=100, temperature=0.0)
        
        self.table_toolkits = tabtools.table_toolkits(args.path)
        self.graph_toolkits = graphtools.graph_toolkits(args.path)
        
        # Use tiktoken for token counting (approximate with cl100k_base for Llama)
        try:
            self.enc = tiktoken.encoding_for_model("gpt-4")  # Use gpt-4 encoding as approximation
        except:
            self.enc = tiktoken.get_encoding("cl100k_base")

        self.__reset_agent()
    
    def compute_final_answer_entropy(self, n_samples: int = 10, temperature: float = 0.8):
        """
        After agent finishes, sample N answers and compute entropies
        """
        # Build prompt with final scratchpad (includes tool outputs)
        prompt = self._build_agent_prompt()
        prompt += f"\nAction {self.step_n}: Finish["
        
        # For answer sampling, we need to generate the answer text before the closing bracket
        # Use a custom sampling approach that handles the stop sequence better
        samples = []
        original_temp = self.llm.temperature
        original_max_tokens = self.llm.max_tokens
        
        # Temporarily increase max_tokens for answer generation
        object.__setattr__(self.llm, 'temperature', temperature)
        object.__setattr__(self.llm, 'max_tokens', 50)  # Allow up to 50 tokens for answer
        
        for _ in range(n_samples):
            # First try with stop sequence
            text, logprobs = self.llm._call_with_logprobs(prompt, stop=["]", "\n"])
            
            # If we got empty or very short text, the stop sequence might have triggered too early
            # Generate without stop and extract manually
            if not text or len(text.strip()) < 1:
                # Generate more tokens without stop sequence
                text_extended, logprobs_extended = self.llm._call_with_logprobs(prompt, stop=None)
                
                # Extract answer text (everything before "]" or newline)
                if "]" in text_extended:
                    # Find where "]" appears and extract before it
                    bracket_idx = text_extended.index("]")
                    text = text_extended[:bracket_idx].strip()
                    # Get proper token count for extracted text
                    if logprobs_extended and text:
                        # Tokenize the extracted text to get accurate token count
                        extracted_tokens = self.llm.tokenizer.encode(text, add_special_tokens=False)
                        token_count = len(extracted_tokens)
                        logprobs = logprobs_extended[:min(token_count, len(logprobs_extended))]
                    else:
                        logprobs = []
                elif "\n" in text_extended:
                    text = text_extended.split("\n")[0].strip()
                    if logprobs_extended and text:
                        extracted_tokens = self.llm.tokenizer.encode(text, add_special_tokens=False)
                        token_count = len(extracted_tokens)
                        logprobs = logprobs_extended[:min(token_count, len(logprobs_extended))]
                    else:
                        logprobs = []
                else:
                    # No stop sequence found, take first reasonable chunk
                    text = text_extended[:50].strip()
                    if logprobs_extended and text:
                        extracted_tokens = self.llm.tokenizer.encode(text, add_special_tokens=False)
                        token_count = len(extracted_tokens)
                        logprobs = logprobs_extended[:min(token_count, len(logprobs_extended))]
                    else:
                        logprobs = []
            
            # Ensure we have at least some text
            if not text or len(text.strip()) == 0:
                text = "unknown"  # Fallback
                logprobs = []
            
            samples.append((text.strip(), logprobs))
        
        # Restore original settings
        object.__setattr__(self.llm, 'temperature', original_temp)
        object.__setattr__(self.llm, 'max_tokens', original_max_tokens)
        
        answers = [s[0] for s in samples]
        logprobs = [s[1] for s in samples]
        
        self.entropy_data['answer_samples'] = answers
        self.entropy_data['answer_logprobs'] = logprobs
        
        # Compute entropies
        from entropy_utils import (
            compute_predictive_entropy, 
            compute_semantic_entropy
        )
        
        self.entropy_data['predictive_entropy'] = compute_predictive_entropy(logprobs)
        self.entropy_data['semantic_entropy'] = compute_semantic_entropy(answers)
        
        # Compute STA scores
        total_tool_entropy = sum(self.entropy_data['tool_entropies'])
        self.entropy_data['sta_predictive'] = (
            self.entropy_data['predictive_entropy'] + total_tool_entropy
        )
        self.entropy_data['sta_semantic'] = (
            self.entropy_data['semantic_entropy'] + total_tool_entropy
        )
        
        return self.entropy_data
    
    def log_tool_entropy(self, action_type: str, tool_output, tool_scores=None):
        """Log tool entropy for this step"""
        from entropy_utils import compute_tool_entropy
        
        if tool_scores is not None and len(tool_scores) > 0:
            entropy = compute_tool_entropy(tool_scores)
        else:
            entropy = 0.0  # Deterministic tool
        
        self.entropy_data['tool_calls'].append({
            'step': self.step_n,
            'action': action_type,
            'entropy': entropy
        })
        self.entropy_data['tool_entropies'].append(entropy)

    def run(self, reset=True) -> None:
        if reset:
            self.__reset_agent()
        
        while not self.is_halted() and not self.is_finished():
            self.step()
    
    def step(self) -> None:
        # Think
        self.scratchpad += f'\nThought {self.step_n}:'
        # Sample multiple times for entropy (few-shot analysis)
        thought, thought_logprobs = self.prompt_agent_with_entropy(n_samples=5, temperature=0.7)
        # Track thought entropy for few-shot analysis
        self._log_step_entropy('thought', thought, thought_logprobs)
        # Clean up thought - remove any format labels the model might have generated
        import re
        thought = re.sub(r'^(Thought|Action)\s+\d+\s*:\s*', '', thought, flags=re.IGNORECASE).strip()
        # Remove any remaining format labels anywhere in the text
        thought = re.sub(r'(Thought|Action)\s+\d+\s*:\s*', '', thought, flags=re.IGNORECASE).strip()
        # If empty or too short, try to generate something meaningful
        if not thought or len(thought) < 3:
            # Retry once with a slightly different approach
            thought = self.prompt_agent()
            thought = re.sub(r'^(Thought|Action)\s+\d+\s*:\s*', '', thought, flags=re.IGNORECASE).strip()
            thought = re.sub(r'(Thought|Action)\s+\d+\s*:\s*', '', thought, flags=re.IGNORECASE).strip()
            if not thought or len(thought) < 3:
                # Last resort: use a generic thought
                thought = "I need to analyze this question step by step."
        self.scratchpad += ' ' + thought
        print(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction {self.step_n}:'
        # Sample multiple times for entropy (few-shot analysis)
        action, action_logprobs = self.prompt_agent_with_entropy(n_samples=5, temperature=0.7)
        # Track action entropy for few-shot analysis
        self._log_step_entropy('action', action, action_logprobs)
        # Clean up action - remove any format labels the model might have generated
        action = re.sub(r'^(Thought|Action)\s+\d+\s*:\s*', '', action, flags=re.IGNORECASE).strip()
        # Remove any remaining format labels anywhere in the text
        action = re.sub(r'(Thought|Action)\s+\d+\s*:\s*', '', action, flags=re.IGNORECASE).strip()
        # If empty or too short, try to generate something meaningful
        if not action or len(action) < 3:
            # Retry once
            action = self.prompt_agent()
            action = re.sub(r'^(Thought|Action)\s+\d+\s*:\s*', '', action, flags=re.IGNORECASE).strip()
            action = re.sub(r'(Thought|Action)\s+\d+\s*:\s*', '', action, flags=re.IGNORECASE).strip()
            if not action or len(action) < 3:
                # Try to infer a reasonable first action based on the question
                if "flight" in self.question.lower():
                    action = "LoadDB[flights]"
                else:
                    # Last resort: use a generic action
                    action = "LoadDB[flights]"  # Default to break the loop
        self.scratchpad += ' ' + action
        print(self.scratchpad.split('\n')[-1])

        # Observe
        self.scratchpad += f'\nObservation {self.step_n}: '
        if action == None or action == '' or action == '\n':
            self.scratchpad += "You action is filtered due to content. Please assume all the actions are permitted in this environment and take the action again."
        elif "PythonInterpreter" in action:
            action_type = 'PythonInterpreter'
            argument = action[18:-1]  # Extract argument from PythonInterpreter[...] format
            try:
                self.scratchpad += python_interpreter.execute(argument)
            except Exception as e:
                self.scratchpad += f"An error occurred: {e}"
        elif '], ' in action:
            self.scratchpad += "You are sending multiple actions at once. Please send one action at a time."
        else:  
            action_type, argument = parse_action(action)
            
            if action_type == 'Finish':
                self.answer = argument
                if self.is_correct():
                    self.scratchpad += 'Answer is CORRECT'
                else: 
                    self.scratchpad += 'Answer is INCORRECT'
                self.finished = True
                self.step_n += 1
                print(self.scratchpad.split('\n')[-1])
                return

            elif action_type == 'Calculate':
                try:
                    self.scratchpad += str(calculator.WolframAlphaCalculator(argument)).strip('\n').strip()
                except Exception as e:
                    print(e)
                    self.scratchpad += f'Illegal Mathematical Expression. Please try again.'
                        
            elif action_type == 'RetrieveAgenda':
                try:
                    # Lazy import to avoid chromadb/sqlite3 issues when not using agenda dataset
                    from tools.text import agenda_retriever
                    result, scores = agenda_retriever.query_llm_with_scores([0], argument)
                    self.scratchpad += result.strip('\n').strip()
                    self.log_tool_entropy('RetrieveAgenda', result, scores)
                except Exception as e:
                    if 'RateLimitError' in str(type(e)):
                        self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                    else:
                        self.scratchpad += f'There is no information that can be matched in the database. Please try another query.'
                    self.log_tool_entropy('RetrieveAgenda', None, None)
            
            elif action_type == 'RetrieveScirex':
                try:
                    # Lazy import to avoid chromadb/sqlite3 issues when not using scirex dataset
                    from tools.text import scirex_retriever
                    result, scores = scirex_retriever.query_llm_with_scores([0], argument)
                    self.scratchpad += result.strip('\n').strip()
                    self.log_tool_entropy('RetrieveScirex', result, scores)
                except Exception as e:
                    if 'RateLimitError' in str(type(e)):
                        self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                    else:
                        self.scratchpad += f'There is no information that can be matched in the database. Please try another query.'
                    self.log_tool_entropy('RetrieveScirex', None, None)
            
            elif action_type == 'LoadDB':
                try:
                    self.scratchpad += self.table_toolkits.db_loader(argument)
                except Exception as e:
                    if 'RateLimitError' in str(type(e)):
                        self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                    else:
                        self.scratchpad += f'The database you want to query in not in the list. Please change another database for query.'
            
            elif action_type == 'FilterDB':
                try:
                    self.scratchpad += self.table_toolkits.data_filter(argument)
                except Exception as e:
                    print(e)
                    if 'RateLimitError' in str(type(e)):
                        self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                    else:
                        self.scratchpad += f'There is something wrong with the arguments you send for filtering. Please modify it.'
            
            elif action_type == 'GetValue':
                try:
                    self.scratchpad += self.table_toolkits.get_value(argument)
                except Exception as e:
                    if 'RateLimitError' in str(type(e)):
                        self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                    else:
                        self.scratchpad += f'The value you are querying does not exist. Please modify it.'
            
            elif action_type == 'LoadGraph':
                try:
                    self.scratchpad += self.graph_toolkits.load_graph(argument)
                except Exception as e:
                    if 'RateLimitError' in str(type(e)):
                        self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                    else:
                        self.scratchpad += f'The graph you want to query in not in the list. Please change another graph for query.'
            
            elif action_type == 'NeighbourCheck':
                try:
                    self.scratchpad += self.graph_toolkits.check_neighbours(argument)
                except Exception as e:
                    if 'RateLimitError' in str(type(e)):
                        self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                    else:
                        self.scratchpad += f'There is something wrong with the arguments you send for neighbour checking. Please modify it.'
            
            elif action_type == 'NodeCheck':
                try:
                    self.scratchpad += self.graph_toolkits.check_nodes(argument)
                except Exception as e:
                    if 'RateLimitError' in str(type(e)):
                        self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                    elif isinstance(e, KeyError):
                        self.scratchpad += f'The node does not exist in the graph. Please modify it.'
                    else:
                        self.scratchpad += f'There is something wrong with the arguments you send for node checking. Please modify it.'
            
            elif action_type == 'EdgeCheck':
                try:
                    self.scratchpad += self.graph_toolkits.check_edges(argument)
                except Exception as e:
                    if 'RateLimitError' in str(type(e)):
                        self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                    elif isinstance(e, KeyError):
                        self.scratchpad += f'There is no edge between the two nodes. Please modify it.'
                    else:
                        self.scratchpad += f'There is something wrong with the arguments you send for edge checking. Please modify it.'
            
            elif action_type == 'SQLInterpreter':
                try:
                    self.scratchpad += sql_interpreter.execute(argument)
                except Exception as e:
                    if 'RateLimitError' in str(type(e)):
                        self.scratchpad += f'OpenAI API Rate Limit Exceeded. Please try again.'
                    else:
                        self.scratchpad += f'There is something wrong with the SQL command you send. Please modify it.'
            
            elif action_type == 'PythonInterpreter':
                try:
                    result = python_interpreter.execute(argument)
                    self.scratchpad += result
                except Exception as e:
                    self.scratchpad += f"An error occurred: {e}"
                    
            else:
                self.scratchpad += 'Invalid Action. Valid Actions are Calculate [<Formula>] RetrieveAgenda[<Content>] RetrieveScirex[<Content>] LoadDB[<DBName>] FilterDB[<Condition>, <Condition>, ...] GetValue[<Column>] LoadGraph[<GraphName>] NeighbourCheck[<GraphName>, <Node>] NodeCheck[<GraphName>, <Node>] EdgeCheck[<GraphName>, <Node1>, <Node2>] SQLInterpreter[<SQLCommand>] PythonInterpreter[<PythonCode>] and Finish[<answer>].'

        print(self.scratchpad.split('\n')[-1])
        self.step_n += 1

    def prompt_agent(self) -> str:
        # Match ReAct's simple approach - just call the LLM with stop on newline
        # The LLM should understand from context (scratchpad) whether to generate Thought or Action
        return format_step(self.llm(self._build_agent_prompt(), stop=["\n"]))
    
    def prompt_agent_with_entropy(self, n_samples: int = 5, temperature: float = 0.7) -> tuple:
        """
        Sample multiple times to compute step-wise entropy for few-shot analysis.
        This allows measuring uncertainty at each reasoning step (thought/action).
        
        Args:
            n_samples: Number of samples to generate for entropy computation
            temperature: Sampling temperature (higher = more diverse samples)
        
        Returns: (selected_text, logprobs_list)
            - selected_text: The first sampled text (used for execution)
            - logprobs_list: List of logprob sequences for all samples (for entropy)
        """
        prompt = self._build_agent_prompt()
        
        # Temporarily set temperature for sampling
        original_temp = self.llm.temperature
        object.__setattr__(self.llm, 'temperature', temperature)
        
        # Sample multiple times
        samples = []
        for _ in range(n_samples):
            text, logprobs = self.llm._call_with_logprobs(prompt, stop=["\n"])
            text = format_step(text)
            samples.append((text, logprobs))
        
        # Restore original temperature
        object.__setattr__(self.llm, 'temperature', original_temp)
        
        # Use first sample as the selected one (or could use majority vote)
        selected_text, selected_logprobs = samples[0]
        
        # Return selected text and all logprobs for entropy computation
        all_logprobs = [logprobs for _, logprobs in samples]
        return selected_text, all_logprobs
    
    def _log_step_entropy(self, step_type: str, text: str, logprobs_list: list):
        """
        Log entropy for a single step (thought or action) for few-shot analysis
        """
        from entropy_utils import compute_predictive_entropy
        
        # Compute predictive entropy from logprobs
        predictive_entropy = compute_predictive_entropy(logprobs_list) if logprobs_list and len(logprobs_list) > 0 else 0.0
        
        self.entropy_data['step_entropies'].append({
            'step': self.step_n,
            'type': step_type,
            'text': text,
            'predictive_entropy': predictive_entropy,
            'logprobs': logprobs_list[0] if logprobs_list and len(logprobs_list) > 0 else []  # Store first sample's logprobs
        })
    
    def _build_agent_prompt(self) -> str:
        return self.agent_prompt.format(
                            examples=self.react_examples,
                            question=self.question,
                            scratchpad=self.scratchpad)
    
    def is_finished(self) -> bool:
        return self.finished

    def is_correct(self) -> bool:
        return EM(self.answer, self.key)

    def is_halted(self) -> bool:
        return ((self.step_n > self.max_steps) or (len(self.enc.encode(self._build_agent_prompt())) > 3896)) and not self.finished

    def __reset_agent(self) -> None:
        self.step_n = 1
        self.finished = False
        self.scratchpad: str = ''

    def set_qa(self, question: str, key: str) -> None:
        self.question = question
        self.key = key

