import numpy as np

def hybrid_scores(content_scores, collab_scores, alpha=0.6):
    """
    Combines content and collaborative scores.
    alpha: Weight for content-based filtering (0.0 to 1.0).
    """
    if collab_scores is None:
        return content_scores
    
    # Weighted average
    return (alpha * content_scores) + ((1 - alpha) * collab_scores)