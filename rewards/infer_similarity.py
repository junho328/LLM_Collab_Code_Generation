"""
Similarity computation module for mental simulation in multi-agent code generation.

Uses GraphCodeBERT embeddings to compute semantic similarity between code snippets.
This is used to measure how well Agent 2's inference matches Agent 1's actual implementation.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Union
from transformers import AutoTokenizer, AutoModel


class CodeSimilarityModel:
    """
    Lazy-loaded GraphCodeBERT model for computing code similarity.
    
    The model is only loaded when first used to avoid unnecessary GPU memory usage
    during training initialization.
    """
    
    _instance = None
    _model = None
    _tokenizer = None
    _device = None
    
    MODEL_NAME = "microsoft/graphcodebert-base"
    
    @classmethod
    def get_instance(cls) -> "CodeSimilarityModel":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def _ensure_loaded(cls):
        """Ensure the model is loaded (lazy loading)."""
        if cls._model is None:
            cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls._tokenizer = AutoTokenizer.from_pretrained(cls.MODEL_NAME)
            cls._model = AutoModel.from_pretrained(cls.MODEL_NAME)
            cls._model.eval()
            cls._model.to(cls._device)
    
    @classmethod
    def encode_code(cls, code: str) -> torch.Tensor:
        """
        Encode a code snippet into a vector embedding.
        
        Args:
            code: The code string to encode.
            
        Returns:
            Tensor of shape (1, 768) containing the CLS token embedding.
        """
        cls._ensure_loaded()
        
        if not code or not isinstance(code, str):
            # Return zero embedding for empty/invalid input
            return torch.zeros(1, 768, device=cls._device)
        
        inputs = cls._tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        inputs = {k: v.to(cls._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = cls._model(**inputs)
        
        # CLS token embedding
        embedding = outputs.last_hidden_state[:, 0]  # (1, 768)
        return embedding
    
    @classmethod
    def encode_batch(cls, codes: List[str]) -> torch.Tensor:
        """
        Encode a batch of code snippets.
        
        Args:
            codes: List of code strings to encode.
            
        Returns:
            Tensor of shape (batch_size, 768) containing embeddings.
        """
        cls._ensure_loaded()
        
        if not codes:
            return torch.zeros(0, 768, device=cls._device)
        
        # Filter and handle empty codes
        valid_codes = [c if c and isinstance(c, str) else "" for c in codes]
        
        inputs = cls._tokenizer(
            valid_codes,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        inputs = {k: v.to(cls._device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = cls._model(**inputs)
        
        # CLS token embeddings for all items in batch
        embeddings = outputs.last_hidden_state[:, 0]  # (batch_size, 768)
        return embeddings


def compute_inference_similarity(
    inference_code: str,
    actual_code: str,
    normalize: bool = True,
) -> float:
    """
    Compute the cosine similarity between Agent 2's inference and Agent 1's actual code.
    
    This function measures how well Agent 2 predicted what Agent 1 would implement.
    
    Args:
        inference_code: Agent 2's inference/prediction of the helper function.
        actual_code: Agent 1's actual implementation of the helper function.
        normalize: Whether to normalize similarity to [0, 1] range (default True).
            If False, returns raw cosine similarity in [-1, 1].
    
    Returns:
        Float similarity score. Higher means better prediction.
        - If normalize=True: Returns value in [0, 1]
        - If normalize=False: Returns raw cosine similarity in [-1, 1]
    """
    model = CodeSimilarityModel.get_instance()
    
    emb_inference = model.encode_code(inference_code)
    emb_actual = model.encode_code(actual_code)
    
    # Compute cosine similarity
    similarity = F.cosine_similarity(emb_inference, emb_actual).item()
    
    if normalize:
        # Normalize from [-1, 1] to [0, 1]
        similarity = (similarity + 1) / 2
    
    return similarity


def compute_batch_inference_similarity(
    inference_codes: List[str],
    actual_codes: List[str],
    normalize: bool = True,
) -> List[float]:
    """
    Compute similarity scores for a batch of inference-actual code pairs.
    
    More efficient than calling compute_inference_similarity multiple times
    as it batches the encoding.
    
    Args:
        inference_codes: List of Agent 2's inferences.
        actual_codes: List of Agent 1's actual implementations.
        normalize: Whether to normalize similarity to [0, 1] range.
    
    Returns:
        List of similarity scores, one per pair.
    """
    if len(inference_codes) != len(actual_codes):
        raise ValueError(
            f"Length mismatch: {len(inference_codes)} inferences vs {len(actual_codes)} actual codes"
        )
    
    if not inference_codes:
        return []
    
    model = CodeSimilarityModel.get_instance()
    
    # Batch encode both sets
    emb_inferences = model.encode_batch(inference_codes)
    emb_actuals = model.encode_batch(actual_codes)
    
    # Compute pairwise cosine similarity
    # Using einsum for efficient batch computation
    # Normalize embeddings first
    emb_inferences_norm = F.normalize(emb_inferences, p=2, dim=1)
    emb_actuals_norm = F.normalize(emb_actuals, p=2, dim=1)
    
    # Element-wise cosine similarity for paired items
    similarities = (emb_inferences_norm * emb_actuals_norm).sum(dim=1)
    
    if normalize:
        # Normalize from [-1, 1] to [0, 1]
        similarities = (similarities + 1) / 2
    
    return similarities.cpu().tolist()


def get_similarity_reward_func(normalize: bool = True):
    """
    Return a callable that can be used as the similarity reward function.
    
    This is a factory function that returns a function compatible with
    the MAGRPOTrainer's similarity_reward_func interface.
    
    Args:
        normalize: Whether to normalize similarity to [0, 1] range.
    
    Returns:
        Callable that takes (inference_code, actual_code) and returns float.
    """
    def similarity_func(inference_code: str, actual_code: str) -> float:
        return compute_inference_similarity(inference_code, actual_code, normalize=normalize)
    
    return similarity_func


# Convenience function for quick testing
def _test_similarity():
    """Test the similarity computation with example code."""
    code_1 = """
def add(a, b):
    return a + b
"""

    code_2 = """
def sum_numbers(x, y):
    return x + y
"""

    similarity = compute_inference_similarity(code_1, code_2)
    print(f"Cosine similarity (normalized): {similarity:.4f}")
    
    similarity_raw = compute_inference_similarity(code_1, code_2, normalize=False)
    print(f"Cosine similarity (raw): {similarity_raw:.4f}")


if __name__ == "__main__":
    _test_similarity()
