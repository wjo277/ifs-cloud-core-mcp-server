# IFS Cloud Semantic Search Implementation

## Complete GPU Training → CPU Production Pipeline

This document provides a comprehensive guide to the semantic search implementation for the IFS Cloud MCP Server. The system is designed to train on GPU for optimal performance and run on CPU in production for cost efficiency.

## 🎯 System Overview

### Architecture Components

1. **Data Structures** (`data_structures.py`)

   - `CodeChunk`: Rich metadata representation of code segments
   - `SearchResult`: Enhanced search results with relevance explanations
   - `IFSMetadataExtractor`: IFS-specific pattern extraction

2. **Data Loading** (`data_loader.py`)

   - `IFSCodeDataset`: Processes entire IFS codebase into training data
   - `ChunkingConfig`: Language-specific code segmentation
   - `DataAugmenter`: Query generation and negative sampling

3. **Neural Models** (`models.py`)

   - `IFSSemanticModel`: Transformer backbone + IFS domain adaptation
   - `IFSSpecificAdapter`: Business terminology and API pattern recognition
   - `MultiTaskLoss`: Combined similarity, classification, and regression training

4. **Training Pipeline** (`trainer.py`)

   - `SemanticTrainer`: GPU-optimized training with mixed precision
   - `TrainingConfig`: Comprehensive hyperparameter management
   - Multi-task learning with contrastive similarity optimization

5. **Vector Indexing** (`indexer.py`)

   - `FAISSIndexManager`: Production-scale similarity search
   - Automatic index optimization based on dataset size
   - CPU/GPU flexibility with memory-efficient configurations

6. **Production Engine** (`engine.py`)
   - `SemanticSearchEngine`: Complete search interface
   - Result caching and performance optimization
   - Rich filtering and specialized search methods

## 🚀 Quick Start Guide

### Prerequisites

```bash
# Core dependencies
pip install torch transformers faiss-cpu numpy

# Optional for advanced features
pip install faiss-gpu  # For GPU indexing
pip install fastai     # For enhanced training
```

### Training Phase (GPU Recommended)

```python
from ifs_cloud_mcp_server.semantic_search import (
    IFSCodeDataset, SemanticTrainer, TrainingConfig
)

# 1. Load IFS codebase
dataset = IFSCodeDataset([Path("_work")])
chunks = list(dataset.load_all_chunks())

# 2. Configure training for GPU
config = TrainingConfig(
    device="cuda",
    batch_size=16,
    num_epochs=10,
    embedding_dim=768,
    output_dir=Path("models/semantic_search")
)

# 3. Train model
trainer = SemanticTrainer(config, chunks)
results = trainer.train()

print(f"Model saved to: {results['model_path']}")
```

### Production Deployment (CPU Optimized)

```python
from ifs_cloud_mcp_server.semantic_search import SemanticSearchEngine

# 1. Initialize search engine
engine = SemanticSearchEngine(
    model_path=Path("models/semantic_search/final_model"),
    device='cpu'  # Production runs on CPU
)

# 2. Initialize and index code
engine.initialize()
engine.index_code_chunks(chunks)

# 3. Search
results = engine.search("create new order", k=10)
for result in results:
    print(f"{result.chunk.function_name}: {result.similarity_score:.3f}")
```

## 📊 Training Methodology

### Multi-Task Learning Approach

Our training combines multiple objectives for robust semantic understanding:

1. **Similarity Learning** (Primary, Weight: 1.0)

   - Contrastive loss between positive/negative code pairs
   - Teaches model to distinguish relevant vs irrelevant code

2. **Module Classification** (Auxiliary, Weight: 0.3)

   - Predicts IFS module (ORDER, INVOICE, INVENTORY, etc.)
   - Improves business domain understanding

3. **Architectural Layer Prediction** (Auxiliary, Weight: 0.2)

   - Classifies code by layer (presentation/business/data/integration)
   - Enhances architectural context awareness

4. **Code Quality Regression** (Auxiliary, Weight: 0.1)
   - Predicts complexity and quality scores
   - Helps rank similar results by code quality

### Data Augmentation Strategy

```python
# Example of generated training pairs
chunk = CodeChunk(
    function_name="Create_Customer_Order",
    module="ORDER",
    business_terms=["customer", "order", "product"],
    api_calls=["Customer_Order_API.New", "Product_API.Get_Info"]
)

# Generated positive queries:
queries = [
    "create new customer order",
    "how to create order",
    "customer order creation function",
    "using Customer_Order_API.New",
    "ORDER module create function"
]
```

### Embedding Text Optimization

Each code chunk is converted to embedding-friendly text that maximizes semantic matching:

```python
def to_embedding_text(chunk):
    parts = [
        f"Function named {chunk.function_name}",
        f"Module: {chunk.module}",
        f"Business concepts: {', '.join(chunk.business_terms)}",
        f"Uses APIs: {', '.join(chunk.api_calls)}",
        f"Architecture layer: {chunk.layer}",
        f"Code pattern: {simplified_code_snippet}"
    ]
    return " | ".join(parts)
```

## 🔧 Configuration Guide

### Training Configuration

```python
config = TrainingConfig(
    # Model Architecture
    model_name="microsoft/codebert-base",  # Pre-trained code model
    embedding_dim=768,                     # Vector size (balance accuracy/speed)
    hidden_dim=256,                        # Internal processing dimension
    num_ifs_classes=50,                    # Number of IFS modules/categories

    # Training Hyperparameters
    batch_size=16,                         # GPU memory dependent
    learning_rate=2e-5,                    # Conservative for fine-tuning
    num_epochs=10,                         # Full training cycles
    warmup_steps=1000,                     # Gradual learning rate increase

    # Hardware Optimization
    mixed_precision=True,                  # Faster training, minimal accuracy loss
    gradient_clip_norm=1.0,               # Training stability
    device="cuda" if available else "cpu"
)
```

### Chunking Configuration

```python
chunking_config = ChunkingConfig(
    min_chunk_size=200,      # Skip trivial code blocks
    max_chunk_size=2000,     # Maintain semantic coherence
    target_chunk_size=800,   # Optimal size for most functions
    overlap_size=100,        # Context preservation between chunks
    min_meaningful_lines=5   # Quality threshold
)
```

### Index Configuration

```python
# Automatic optimization based on dataset size
config = optimize_index_config(
    dataset_size=len(chunks),
    embedding_dim=768,
    memory_limit_gb=4.0
)

# Manual configuration for specific needs
config = IndexConfig(
    approximate_search=True,    # Use for datasets > 10k
    n_clusters=1000,           # Balance accuracy/speed
    n_probe=50,                # Search thoroughness
    use_pq=True,              # Memory compression for large datasets
    use_gpu=False             # Production typically uses CPU
)
```

## 📈 Performance Optimization

### GPU Training Performance

- **Mixed Precision**: 30-50% faster training with minimal accuracy loss
- **Gradient Accumulation**: Simulate larger batches on limited GPU memory
- **Learning Rate Scheduling**: Warm-up + cosine decay for optimal convergence
- **Multi-GPU Support**: Distributed training for very large datasets

### CPU Inference Performance

- **Model Quantization**: Reduce model size by 4x with minimal accuracy loss
- **FAISS Optimization**: Automatic index selection based on dataset characteristics
- **Result Caching**: In-memory LRU cache for frequent queries
- **Batch Processing**: Efficient embedding generation for multiple queries

### Memory Management

```python
# Memory-efficient batch processing
def process_large_dataset(chunks, batch_size=1000):
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        embeddings = model.encode_text([c.to_embedding_text() for c in batch])
        yield embeddings, batch
```

## 🎯 Production Integration

### Search Engine Integration

```python
class IFSSearchService:
    def __init__(self):
        self.semantic_engine = SemanticSearchEngine(
            model_path=Path("models/semantic_search/final_model"),
            device='cpu'
        )
        self.semantic_engine.initialize()

    async def search(self, query: str, filters: dict = None):
        # Combine with existing search for hybrid results
        semantic_results = self.semantic_engine.search(query, k=20)
        traditional_results = self.existing_search_engine.search(query)

        # Merge and rank results
        return self.merge_search_results(semantic_results, traditional_results)
```

### MCP Server Integration

```python
@server.call_tool()
async def semantic_search(query: str, max_results: int = 10):
    """Enhanced semantic search tool."""
    results = await search_service.semantic_search(query, max_results)

    return [
        {
            'function_name': r.chunk.function_name,
            'file_path': r.chunk.file_path,
            'similarity_score': r.similarity_score,
            'relevance_explanation': r.relevance_explanation,
            'module': r.chunk.module,
            'business_terms': r.chunk.business_terms,
            'code_preview': r.chunk.raw_content[:500]
        }
        for r in results
    ]
```

## 🔍 Advanced Search Features

### Specialized Search Methods

```python
# Find similar functions
similar_functions = engine.suggest_similar_functions(
    function_name="Create_Customer_Order",
    module="ORDER"
)

# Find API usage examples
api_examples = engine.find_usage_examples(
    api_call="Customer_Order_API.New",
    module="ORDER"
)

# Business context search
business_examples = engine.get_business_context_examples(
    business_term="customer order processing"
)
```

### Advanced Filtering

```python
results = engine.search(
    query="create customer order",
    filters={
        'module': ['ORDER', 'CUSTOMER'],           # Multiple modules
        'layer': 'business',                       # Specific layer
        'file_type': ['.plsql', '.sql'],          # File types
        'has_error_handling': True,                # Quality requirements
        'max_complexity': 0.7                      # Complexity threshold
    }
)
```

## 📊 Monitoring and Analytics

### Training Metrics

```python
# Training progress monitoring
{
    'epoch': 5,
    'train_loss': 0.234,
    'val_loss': 0.267,
    'similarity_loss': 0.180,
    'classification_accuracy': 0.847,
    'learning_rate': 1.5e-5
}
```

### Production Metrics

```python
# Search performance monitoring
{
    'total_queries': 15420,
    'average_search_time': 0.023,      # seconds
    'cache_hit_rate': 0.73,            # 73% cache hits
    'index_size': 2847293,             # vectors
    'memory_usage': '1.2GB'
}
```

### Quality Metrics

- **Semantic Accuracy**: Manual evaluation on held-out queries
- **Business Relevance**: Domain expert assessment of results
- **Response Time**: P95 latency under production load
- **Cache Efficiency**: Hit rates and memory utilization

## 🛠️ Troubleshooting Guide

### Common Training Issues

**GPU Out of Memory**

```python
# Solutions:
config.batch_size = 8        # Reduce batch size
config.mixed_precision = True  # Enable AMP
config.gradient_accumulation_steps = 4  # Simulate larger batches
```

**Poor Convergence**

```python
# Solutions:
config.learning_rate = 1e-5   # Lower learning rate
config.warmup_steps = 2000    # More gradual warmup
config.weight_decay = 0.01    # Add regularization
```

**Overfitting**

```python
# Solutions:
config.dropout_rate = 0.2     # Increase dropout
config.validation_ratio = 0.2  # More validation data
config.early_stopping = True   # Stop when val loss increases
```

### Production Issues

**Slow Search Performance**

```python
# Solutions:
config.approximate_search = True    # Use approximate index
config.n_probe = 20                # Reduce search thoroughness
engine.clear_cache()               # Clear memory
```

**Poor Search Quality**

```python
# Solutions:
# Retrain with more diverse queries
# Adjust embedding text generation
# Increase min_score threshold
# Add domain-specific fine-tuning
```

## 📚 API Reference

### Core Classes

#### `SemanticSearchEngine`

- `initialize()`: Set up model and index
- `search(query, k, filters)`: Main search interface
- `batch_search(queries)`: Efficient multi-query search
- `index_code_chunks(chunks)`: Build search index

#### `SemanticTrainer`

- `train()`: Execute complete training pipeline
- `save_checkpoint()`: Save training state
- `validate_epoch()`: Run validation loop

#### `IFSSemanticModel`

- `encode_text(texts)`: Generate embeddings
- `save_model(path)`: Persist for production
- `load_model(path)`: Restore saved model

## 🔮 Future Enhancements

### Planned Improvements

1. **Incremental Learning**: Update model with new code without full retraining
2. **Multi-Language Support**: Extend beyond current IFS languages
3. **Graph-Based Search**: Incorporate code dependency relationships
4. **Auto-Documentation**: Generate explanations for search results
5. **A/B Testing Framework**: Compare search algorithm variants

### Research Directions

1. **Code Clone Detection**: Find duplicate/similar code patterns
2. **Semantic Code Generation**: Generate code from natural language
3. **Impact Analysis**: Predict change effects using semantic similarity
4. **Automated Refactoring**: Suggest improvements based on patterns

## 📝 Contributing

### Development Setup

```bash
git clone <repository>
cd ifs-cloud-core-mcp-server

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/semantic_search/

# Train test model
python train_semantic_search.py --data-dir "_work" --quick-test
```

### Code Standards

- All new features require comprehensive documentation
- Include usage examples in docstrings
- AI agents should understand the reasoning behind design choices
- Performance implications must be documented
- Error handling should be robust and informative

---

This implementation provides a production-ready semantic search system that bridges the gap between natural language queries and IFS enterprise code, enabling more intuitive and powerful code discovery for both human developers and AI agents.
