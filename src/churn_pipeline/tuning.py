"""Training strategies: tuning and cross-validation (implemented in Stage 5).

Encapsulates the three training paths (search, plain CV, single fit) so preprocessing is
re-fit inside every CV fold and the held-out test set is evaluated exactly once.
"""
