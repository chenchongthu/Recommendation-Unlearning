try:
    from .cpp.evaluate_foldout import eval_score_matrix_foldout
except Exception:
    from .python.evaluate_foldout import eval_score_matrix_foldout
