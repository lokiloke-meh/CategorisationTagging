import numpy as np
from bisect import bisect
from evals.base_evals import BaseEvaluator

class EmbeddingsEvaluator(BaseEvaluator):

    @staticmethod
    def calc_cosine_similarity(vector_array_1:list[float], vector_array_2:list[float]):
        dot_product = np.inner(vector_array_1, vector_array_2)
        
        # Handle cases where one or both magnitudes might be zero to prevent division by zero.
        # If either vector is a zero vector, cosine similarity is undefined or 0.
        # For practical purposes in embeddings, it's often treated as 0 or a very small number.
        # Here, we'll return 0 if either magnitude is zero.
        magnitude_a = np.linalg.norm(vector_array_1)
        magnitude_b = np.linalg.norm(vector_array_2)
        
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        cosine_similarity = dot_product / (magnitude_a * magnitude_b)

        # .item() extracts the value and converts numpy.float64 to a clean Python float
        return float(cosine_similarity.item()) if hasattr(cosine_similarity, 'item') else float(cosine_similarity)
    
    @staticmethod
    def score_cosine_similarity_result(cosine_similarity:float):
        """
        Maps a continuous cosine similarity score to a discrete set of evaluation scores.
        
        The scoring logic is as follows:
        - If cosine_similarity <= 0.5, the original cosine_similarity is returned.
        - If 0.5 < cosine_similarity <= 0.8, a score of 0.75 is returned.
        - If 0.8 < cosine_similarity <= 1.0, a score of 1.0 is returned.
        
        Args:
            cosine_similarity (float): The calculated cosine similarity score, typically between -1.0 and 1.0.
            
        Returns:
            float: The mapped discrete score (either the original similarity, 0.75, or 1.0).
        """
        
        scoring_breakpoints = [0.5, 0.8, 1]
        # The first element of scoring_buckets is special: if cosine_similarity is below the first breakpoint,
        # the original cosine_similarity value is returned. Otherwise, fixed scores are used.
        scoring_buckets = [cosine_similarity, 0.75, 1.0]
        
        scoring_bucket_index = bisect(scoring_breakpoints, cosine_similarity)
        return scoring_buckets[scoring_bucket_index]
