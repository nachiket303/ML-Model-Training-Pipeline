"""Feature preprocessing (implemented in Stage 3).

Builds an unfitted scikit-learn ColumnTransformer (numeric impute+scale, categorical
impute+one-hot) so preprocessing is fitted on training data only and reused consistently.
"""
