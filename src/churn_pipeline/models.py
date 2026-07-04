"""Model factory (implemented in Stage 3).

Maps a config model name + params to an unfitted scikit-learn estimator, injecting the
global seed so every model is reproducible.
"""
