"""
Complete Semantic Search Training Script
========================================

This script demonstrates the complete workflow for training a semantic search model
on GPU and deploying it for CPU inference. This is a comprehensive example showing
how to train on IFS codebase and create a production-ready search engine.

Usage:
    # Full training pipeline
    python train_semantic_search.py --data-dir "_work" --output-dir "models/semantic" --gpu

    # CPU inference testing
    python train_semantic_search.py --model-path "models/semantic" --test-search --cpu

    # Quick test with small dataset
    python train_semantic_search.py --data-dir "_work" --quick-test
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional
import json
import gc

# GPU memory management
import torch

if torch.cuda.is_available():
    # Clear any existing GPU memory
    torch.cuda.empty_cache()
    gc.collect()
    # Set memory optimization settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

# Set up the path to import our modules
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.ifs_cloud_mcp_server.semantic_search import (
    # Data processing
    IFSCodeDataset,
    ChunkingConfig,
    DataAugmenter,
    # Model and training
    IFSSemanticModel,
    SemanticTrainer,
    TrainingConfig,
    create_train_val_split,
    # Production inference
    SemanticSearchEngine,
    FAISSIndexManager,
    IndexConfig,
    optimize_index_config,
    # Data structures
    CodeChunk,
    SearchResult,
)


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None):
    """Setup logging configuration."""
    level = getattr(logging, log_level.upper())

    handlers = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def load_ifs_codebase(
    data_dirs: List[Path], cache_name: str = "ifs_training"
) -> List[CodeChunk]:
    """
    Load and process IFS codebase into training chunks.

    CODEBASE LOADING STRATEGY:
    -------------------------
    1. Configure chunking for optimal semantic units
    2. Process all code files in parallel
    3. Extract rich metadata for each chunk
    4. Cache processed chunks for faster iteration
    5. Return chunks ready for training
    """
    logging.info("=" * 60)
    logging.info("LOADING IFS CODEBASE FOR TRAINING")
    logging.info("=" * 60)

    # Configure chunking for IFS code
    chunking_config = ChunkingConfig(
        min_chunk_size=150,  # Slightly smaller for more granular chunks
        max_chunk_size=1500,  # Reasonable upper bound for semantic coherence
        target_chunk_size=600,  # Sweet spot for most functions
        overlap_size=50,  # Some overlap for context preservation
        min_meaningful_lines=3,  # Skip trivial code blocks
    )

    # Create dataset
    dataset = IFSCodeDataset(
        source_directories=data_dirs,
        config=chunking_config,
        cache_dir=Path("cache/semantic_search"),
    )

    # Try to load from cache first
    cached_chunks = dataset.load_chunks_from_cache(cache_name)
    if cached_chunks:
        logging.info(f"Loaded {len(cached_chunks)} chunks from cache")
        return cached_chunks

    # Load and process all chunks
    logging.info("Processing codebase (this may take several minutes)...")
    start_time = time.time()

    chunks = list(dataset.load_all_chunks())

    loading_time = time.time() - start_time

    # Save to cache for future runs
    dataset.save_chunks_to_cache(chunks, cache_name)

    # Print statistics
    stats = dataset.get_statistics()
    logging.info(f"\nCodebase processing completed in {loading_time/60:.1f} minutes")
    logging.info(f"Total chunks created: {len(chunks)}")
    logging.info(f"Files processed: {stats['files_processed']}")
    logging.info(f"Files skipped: {stats['files_skipped']}")

    # Show language distribution
    if stats["languages"]:
        logging.info("\nLanguage distribution:")
        for lang, count in stats["languages"].items():
            logging.info(f"  {lang}: {count} files")

    # Show chunk size statistics
    if stats.get("chunk_size_stats"):
        size_stats = stats["chunk_size_stats"]
        logging.info(f"\nChunk size statistics:")
        logging.info(f"  Mean size: {size_stats['mean']:.0f} characters")
        logging.info(f"  Min size: {size_stats['min']} characters")
        logging.info(f"  Max size: {size_stats['max']} characters")

    return chunks


def train_semantic_model(
    chunks: List[CodeChunk], output_dir: Path, quick_test: bool = False
) -> Path:
    """
    Train the semantic search model on GPU.

    TRAINING PIPELINE:
    -----------------
    1. Split data into train/validation sets
    2. Configure training for GPU or CPU
    3. Initialize model and trainer
    4. Run training with monitoring
    5. Save model for CPU inference
    6. Return path to trained model
    """
    logging.info("=" * 60)
    logging.info("TRAINING SEMANTIC SEARCH MODEL")
    logging.info("=" * 60)

    # Create train/validation split
    train_chunks, val_chunks = create_train_val_split(
        chunks,
        val_ratio=0.15,  # 15% validation
        stratify_by="module",  # Ensure all modules represented
    )

    # Quick test mode for development
    if quick_test:
        logging.info("Quick test mode: using subset of data")
        train_chunks = train_chunks[: min(100, len(train_chunks))]
        val_chunks = val_chunks[: min(20, len(val_chunks))]

    # Configure training
    config = TrainingConfig(
        # Model architecture
        model_name="microsoft/codebert-base",  # Pre-trained code model
        embedding_dim=384,  # Balanced size for production
        hidden_dim=128,  # Efficient hidden layers
        num_ifs_classes=50,  # IFS module/domain classification
        # Training hyperparameters
        batch_size=16,  # Conservative batch size for stable training
        learning_rate=2e-5,  # Conservative for fine-tuning
        num_epochs=2 if quick_test else 8,  # Quick test vs full training
        warmup_steps=500,  # Gradual learning rate increase
        # Loss weighting (emphasize similarity learning)
        similarity_weight=1.0,  # Primary objective
        classification_weight=0.2,  # Auxiliary task
        regression_weight=0.1,  # Quality prediction
        # Training efficiency
        mixed_precision=True,  # Faster training with minimal loss
        gradient_clip_norm=1.0,  # Stable training
        # Output configuration
        output_dir=output_dir,
        checkpoint_dir=output_dir / "checkpoints",
        log_dir=output_dir / "logs",
    )

    logging.info(f"Training configuration:")
    logging.info(f"  Model: {config.model_name}")
    logging.info(f"  Device: {config.device}")
    logging.info(f"  Batch size: {config.batch_size}")
    logging.info(f"  Epochs: {config.num_epochs}")
    logging.info(f"  Train chunks: {len(train_chunks)}")
    logging.info(f"  Val chunks: {len(val_chunks)}")

    # Create trainer
    trainer = SemanticTrainer(config, train_chunks, val_chunks)

    # Train model
    start_time = time.time()
    results = trainer.train()
    training_time = time.time() - start_time

    # Log training results
    logging.info(f"\nTraining completed in {training_time/3600:.2f} hours")
    logging.info(f"Final model saved to: {results['model_path']}")

    return results["model_path"]


def create_production_index(
    model_path: Path, chunks: List[CodeChunk], index_path: Path
) -> FAISSIndexManager:
    """
    Create FAISS index for production search.

    INDEXING STRATEGY:
    -----------------
    1. Load trained model for inference
    2. Generate embeddings for all chunks
    3. Optimize FAISS index configuration
    4. Build and save index for production
    """
    logging.info("=" * 60)
    logging.info("CREATING PRODUCTION SEARCH INDEX")
    logging.info("=" * 60)

    # Load trained model for CPU inference
    model = IFSSemanticModel.load_model(model_path, device="cpu")
    logging.info(f"Loaded model with {model.embedding_dim}D embeddings")

    # Generate embeddings for all chunks
    logging.info("Generating embeddings...")
    start_time = time.time()

    # Process in batches for memory efficiency
    batch_size = 100
    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_texts = [chunk.to_embedding_text() for chunk in batch_chunks]

        batch_embeddings = model.encode_text(batch_texts, device="cpu")
        all_embeddings.append(batch_embeddings)

        if (i + batch_size) % 1000 == 0:
            logging.info(
                f"Generated embeddings for {i + batch_size}/{len(chunks)} chunks"
            )

    embeddings = np.vstack(all_embeddings)
    embedding_time = time.time() - start_time

    logging.info(f"Embedding generation completed in {embedding_time:.1f} seconds")

    # Optimize index configuration for dataset
    config = optimize_index_config(
        dataset_size=len(chunks),
        embedding_dim=model.embedding_dim,
        memory_limit_gb=2.0,  # Conservative memory limit
    )

    # Create index manager
    index_manager = FAISSIndexManager(
        embedding_dim=model.embedding_dim, config=config, index_dir=index_path.parent
    )

    # Build index
    if not index_manager.create_index():
        raise RuntimeError("Failed to create FAISS index")

    # Add embeddings
    if not index_manager.add_embeddings(embeddings, chunks):
        raise RuntimeError("Failed to add embeddings to index")

    # Save index
    if not index_manager.save_index(index_path.stem):
        logging.warning("Failed to save index to disk")

    logging.info(f"Index created with {index_manager.total_vectors} vectors")

    return index_manager


def test_search_engine(
    model_path: Path, index_path: Path, test_queries: Optional[List[str]] = None
):
    """
    Test the complete search engine with sample queries.

    This demonstrates the production inference pipeline:
    CPU model loading, FAISS search, and result ranking.
    """
    logging.info("=" * 60)
    logging.info("TESTING PRODUCTION SEARCH ENGINE")
    logging.info("=" * 60)

    # Create search engine
    engine = SemanticSearchEngine(
        model_path=model_path,
        index_path=index_path,
        device="cpu",  # Production deployment on CPU
    )

    # Initialize engine
    if not engine.initialize():
        raise RuntimeError("Failed to initialize search engine")

    logging.info("Search engine initialized successfully")

    # Test queries
    if not test_queries:
        test_queries = [
            "create new order",
            "update inventory levels",
            "validate payment information",
            "calculate tax amounts",
            "generate financial reports",
            "handle customer complaints",
            "process invoice payments",
            "manage supplier contracts",
            "track shipment status",
            "audit trail functionality",
        ]

    # Test individual searches
    logging.info("\nTesting individual searches:")
    for query in test_queries[:5]:  # Test first 5 queries
        start_time = time.time()
        results = engine.search(query, k=5, min_score=0.1)
        search_time = time.time() - start_time

        logging.info(f"\nQuery: '{query}'")
        logging.info(f"Results: {len(results)} matches in {search_time*1000:.1f}ms")

        for i, result in enumerate(results[:3], 1):  # Show top 3
            logging.info(
                f"  {i}. {result.chunk.function_name or 'unnamed'} "
                f"(score: {result.similarity_score:.3f})"
            )
            logging.info(
                f"     Module: {result.chunk.module}, " f"Layer: {result.chunk.layer}"
            )
            logging.info(f"     File: {result.chunk.file_path}")

    # Test batch search
    logging.info(f"\nTesting batch search with {len(test_queries)} queries...")
    start_time = time.time()
    batch_results = engine.batch_search(test_queries, k=3)
    batch_time = time.time() - start_time

    total_results = sum(len(results) for results in batch_results)
    logging.info(
        f"Batch search: {total_results} total results in {batch_time*1000:.1f}ms"
    )
    logging.info(f"Average: {batch_time/len(test_queries)*1000:.1f}ms per query")

    # Show engine statistics
    stats = engine.get_statistics()
    logging.info(f"\nEngine Statistics:")
    logging.info(f"  Total queries: {stats['performance']['total_queries']}")
    logging.info(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
    logging.info(f"  Index vectors: {stats['index']['total_vectors']}")

    return engine


def main():
    """Main training and testing pipeline."""
    parser = argparse.ArgumentParser(
        description="Train and deploy IFS semantic search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full training pipeline
    python train_semantic_search.py --data-dir "_work" --output-dir "models/semantic"
    
    # Quick test for development
    python train_semantic_search.py --data-dir "_work" --quick-test
    
    # Test existing model
    python train_semantic_search.py --model-path "models/semantic/final_model" --test-only
        """,
    )

    # Data and model paths
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing IFS code (_work)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/semantic_search"),
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--model-path", type=Path, help="Path to existing model (for testing only)"
    )

    # Training options
    parser.add_argument(
        "--quick-test", action="store_true", help="Quick test with small dataset"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only test existing model, skip training",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining even if model exists",
    )

    # Hardware options
    parser.add_argument(
        "--gpu", action="store_true", help="Force GPU usage for training"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU usage for everything"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument("--log-file", type=Path, help="Log file path")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, args.log_file)

    # GPU memory management for training
    if torch.cuda.is_available() and not args.test_only:
        logging.info("Initializing GPU memory management...")
        torch.cuda.empty_cache()
        gc.collect()
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logging.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB total memory)")

        # Check current memory usage
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        logging.info(
            f"GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
        )

    try:
        # Print banner
        logging.info("=" * 80)
        logging.info("IFS CLOUD SEMANTIC SEARCH TRAINING PIPELINE")
        logging.info("=" * 80)
        logging.info(f"Data directory: {args.data_dir}")
        logging.info(f"Output directory: {args.output_dir}")
        logging.info(f"Quick test mode: {args.quick_test}")
        logging.info(f"Test only: {args.test_only}")

        model_path = args.model_path or (args.output_dir / "final_model")
        index_path = args.output_dir / "search_index.faiss"

        # Step 1: Load IFS codebase (unless testing existing model)
        if not args.test_only:
            if not args.data_dir.exists():
                logging.error(f"Data directory not found: {args.data_dir}")
                return 1

            chunks = load_ifs_codebase([args.data_dir])

            if not chunks:
                logging.error("No code chunks loaded - check data directory")
                return 1

        # Step 2: Train model (unless testing existing model)
        if not args.test_only and (args.force_retrain or not model_path.exists()):
            args.output_dir.mkdir(parents=True, exist_ok=True)
            model_path = train_semantic_model(chunks, args.output_dir, args.quick_test)
        elif not args.test_only:
            logging.info(f"Using existing model: {model_path}")

        # Step 3: Create production index (unless testing existing model)
        if not args.test_only:
            create_production_index(model_path, chunks, index_path)

        # Step 4: Test the complete pipeline
        if model_path.exists():
            test_search_engine(model_path, index_path)
        else:
            logging.error(f"Model not found: {model_path}")
            return 1

        logging.info("=" * 80)
        logging.info("SEMANTIC SEARCH PIPELINE COMPLETED SUCCESSFULLY")
        logging.info("=" * 80)
        logging.info(f"Model available at: {model_path}")
        logging.info(f"Index available at: {index_path}")
        logging.info("")
        logging.info("Integration instructions:")
        logging.info("1. Copy model and index to production server")
        logging.info("2. Use SemanticSearchEngine class for searches")
        logging.info("3. Model runs on CPU, no GPU required in production")

        return 0

    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        return 1
    except Exception as e:
        logging.error(f"Training failed: {e}")
        logging.exception("Full error details:")
        return 1


if __name__ == "__main__":
    import numpy as np  # Add required imports

    exit(main())
