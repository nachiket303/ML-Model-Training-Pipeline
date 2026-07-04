"""Pipeline orchestration (implemented in Stage 4).

The single orchestrator that wires the components together: load config, load and validate
data, split, preprocess, train/evaluate each model, register artifacts, and compare.
"""
