"""Data ingestion and validation layer (implemented in Stage 2).

Loads the raw dataset, validates it at the boundary, applies deliberate data-quality
fixes (e.g. TotalCharges coercion), drops identifier columns, and encodes the target.
"""
