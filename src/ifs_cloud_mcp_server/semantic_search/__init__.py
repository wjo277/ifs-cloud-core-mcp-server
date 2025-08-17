"""
IFS Cloud Semantic Search Module
================================

This module implements a complete semantic search system for IFS enterprise code.
It provides GPU-accelerated training and CPU-optimized inference for production deployment.

SYSTEM ARCHITECTURE:
-------------------
1. DATA_STRUCTURES: Core data types (CodeChunk, SearchResult, metadata)
2. DATA_LOADER: Code chunking, preprocessing, and training data preparation
3. MODELS: Neural network architectures with pre-trained backbone + IFS adaptation
4. TRAINER: GPU training pipeline with multi-task learning and optimization
5. INDEXER: FAISS-based vector similarity search for production inference
6. ENGINE: Production search interface with caching and result ranking

PRODUCTION WORKFLOW:
-------------------
Training Phase (GPU):
1. Load IFS codebase using IFSCodeDataset
2. Train IFSSemanticModel with SemanticTrainer
3. Export trained model for CPU inference

Production Phase (CPU):
1. Load trained model in SemanticSearchEngine
2. Build FAISS index from code embeddings
3. Serve real-time semantic search queries

Example Usage:
-------------
Training:
```python
from ifs_cloud_mcp_server.semantic_search import (
    IFSCodeDataset, SemanticTrainer, TrainingConfig
)

# Load and process code
dataset = IFSCodeDataset([Path("_work")])
chunks = list(dataset.load_all_chunks())

# Train model
config = TrainingConfig(device="cuda", num_epochs=10)
trainer = SemanticTrainer(config, chunks)
results = trainer.train()
```

Production:
```python
from ifs_cloud_mcp_server.semantic_search import SemanticSearchEngine

# Initialize engine
engine = SemanticSearchEngine(
    model_path=Path("models/semantic_search/final_model"),
    index_path=Path("indexes/ifs_code.faiss")
)
engine.initialize()

# Index code for search
engine.index_code_chunks(chunks)

# Search
results = engine.search("how to create new order", k=10)
```
"""

# Core data structures
from .data_structures import CodeChunk, SearchResult, IFSMetadataExtractor

# Data loading and preprocessing
from .data_loader import IFSCodeDataset, ChunkingConfig, DataAugmenter

# Model architectures
from .models import (
    IFSSemanticModel,
    IFSSpecificAdapter,
    ContrastiveLoss,
    MultiTaskLoss,
    count_parameters,
    freeze_layers,
    unfreeze_layers,
)

# Training pipeline
from .trainer import (
    SemanticTrainer,
    TrainingConfig,
    SemanticSearchDataset,
    create_train_val_split,
)

# Vector indexing
from .indexer import FAISSIndexManager, IndexConfig, optimize_index_config

# Production search engine
from .engine import SemanticSearchEngine, create_semantic_engine

# Version information
__version__ = "1.0.0"
__author__ = "IFS Cloud MCP Server Team"

# Main exports for common usage patterns
__all__ = [
    # Data structures
    "CodeChunk",
    "SearchResult",
    "IFSMetadataExtractor",
    # Data loading
    "IFSCodeDataset",
    "ChunkingConfig",
    "DataAugmenter",
    # Models
    "IFSSemanticModel",
    "IFSSpecificAdapter",
    "ContrastiveLoss",
    "MultiTaskLoss",
    # Training
    "SemanticTrainer",
    "TrainingConfig",
    "SemanticSearchDataset",
    "create_train_val_split",
    # Indexing
    "FAISSIndexManager",
    "IndexConfig",
    "optimize_index_config",
    # Engine
    "SemanticSearchEngine",
    "create_semantic_engine",
    # Utilities
    "count_parameters",
    "freeze_layers",
    "unfreeze_layers",
]
