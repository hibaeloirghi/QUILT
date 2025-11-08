import numpy as np
from typing import List, Tuple
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

def compute_predictive_entropy(logprobs_list: List[List[float]]) -> float:
    """
    Compute predictive entropy H(Y|Z,x) from N sampled sequences
    
    Args:
        logprobs_list: List of per-token log probabilities for each sample
    
    Returns:
        Predictive entropy (average negative log probability)
    """
    sequence_logprobs = []
    for logprobs in logprobs_list:
        # Sum log probs across tokens in sequence
        seq_logprob = sum(logprobs)
        sequence_logprobs.append(seq_logprob)
    
    # Monte Carlo estimate: -1/N * sum of log p(y_i | z, x)
    entropy = -np.mean(sequence_logprobs)
    return entropy

def compute_semantic_entropy(answers: List[str], threshold: float = 0.5) -> float:
    """
    Compute semantic entropy H_c(Y|Z,x) by clustering answers
    
    Args:
        answers: List of sampled answer strings
        threshold: Similarity threshold for clustering
    
    Returns:
        Semantic entropy over answer classes
    """
    if len(answers) == 0:
        return 0.0
    
    # Simple string-based clustering for short answers (numbers, names)
    normalized_answers = [normalize_answer(ans) for ans in answers]
    
    if all(ans.replace('.', '').replace('-', '').isdigit() or len(ans.split()) <= 3 
           for ans in normalized_answers):
        # Use exact match for short/numeric answers
        counter = Counter(normalized_answers)
        probs = np.array(list(counter.values())) / len(answers)
        return -np.sum(probs * np.log(probs + 1e-10))
    
    # For longer text, use semantic clustering
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(answers)
    
    # Agglomerative clustering
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=threshold,
        linkage='average'
    )
    labels = clustering.fit_predict(embeddings)
    
    # Compute entropy over clusters
    counter = Counter(labels)
    probs = np.array(list(counter.values())) / len(answers)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    return entropy

def normalize_answer(s: str) -> str:
    """Normalize answer string for comparison"""
    import re
    import string
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the|USD)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_tool_entropy(tool_scores: np.ndarray, top_k: int = None) -> float:
    """
    Compute tool entropy H(Z|a) from tool output distribution
    
    Args:
        tool_scores: Array of scores/logits from tool (e.g., retriever scores)
        top_k: If provided, only consider top-K candidates
    
    Returns:
        Entropy of tool output distribution
    """
    if tool_scores is None or len(tool_scores) == 0:
        return 0.0  # Deterministic tool
    
    if top_k is not None:
        # Get top-K scores
        top_indices = np.argsort(tool_scores)[-top_k:]
        tool_scores = tool_scores[top_indices]
    
    # Convert to probabilities via softmax
    probs = np.exp(tool_scores - np.max(tool_scores))
    probs = probs / np.sum(probs)
    
    # Compute entropy
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    return entropy
