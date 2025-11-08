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
    def __init__(self, pipe, tokenizer, max_tokens=100, temperature=0.0):
        super().__init__()
        self.pipe = pipe
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model = pipe.model  # Store model for direct access
    
    def _call_with_logprobs(self, prompt: str, stop: Optional[List[str]] = None) -> tuple:
        """Call model and return both text and per-token log probabilities"""
        messages = [{"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature if self.temperature > 0 else 1.0,
                do_sample=self.temperature > 0,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Get generated tokens and compute log probs
        generated_ids = outputs.sequences[0][input_ids.shape[1]:]
        logprobs = []
        
        for i, token_id in enumerate(generated_ids):
            if i < len(outputs.scores):
                logits = outputs.scores[i][0]
                log_probs = torch.log_softmax(logits, dim=-1)
                logprobs.append(log_probs[token_id].item())
        
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Handle stop sequences
        if stop:
            for stop_seq in stop:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0]
        
        return generated_text.strip(), logprobs
    
    def sample_answers(self, prompt: str, n_samples: int = 10, 
                      temperature: float = 0.8, stop: Optional[List[str]] = None) -> List[tuple]:
        """Sample N answers with their log probabilities"""
        samples = []
        original_temp = self.temperature
        self.temperature = temperature
        
        for _ in range(n_samples):
            text, logprobs = self._call_with_logprobs(prompt, stop)
            samples.append((text, logprobs))
        
        self.temperature = original_temp
        return samples
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call the model with the given prompt"""
        text, _ = self._call_with_logprobs(prompt, stop)
        return text
    
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
                 ) -> None:
        
        # Add entropy tracking
        self.entropy_data = {
            'tool_calls': [],
            'tool_entropies': [],
            'answer_samples': [],
            'answer_logprobs': [],
            'predictive_entropy': None,
            'semantic_entropy': None,
            'sta_predictive': None,
            'sta_semantic': None
        }
        
        self.question = question
        self.answer = ''
        self.key = key
        self.max_steps = max_steps
        self.agent_prompt = agent_prompt
        if args.prompt == "easy":
            self.react_examples = TOOLQA_EASY8
        else:
            self.react_examples = TOOLQA_HARD3

        # Setup Llama model
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
        
        # Sample N answers
        samples = self.llm.sample_answers(
            prompt, 
            n_samples=n_samples,
            temperature=temperature,
            stop=["]", "\n"]
        )
        
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
        thought = self.prompt_agent()
        self.scratchpad += ' ' + thought
        print(self.scratchpad.split('\n')[-1])

        # Act
        self.scratchpad += f'\nAction {self.step_n}:'
        action = self.prompt_agent()
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
        return format_step(self.llm(self._build_agent_prompt(), stop=["\n"]))
    
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

