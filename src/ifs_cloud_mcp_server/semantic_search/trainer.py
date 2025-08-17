"""
Training Module for IFS Semantic Search
=======================================

This module handles the training pipeline for the semantic search model.
It implements GPU-accelerated training with the ability to export models
for CPU inference in production.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
import numpy as np
import logging
import json
import time
import gc
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from transformers import get_linear_schedule_with_warmup

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from .models import IFSSemanticModel, MultiTaskLoss, count_parameters
from .data_structures import CodeChunk, SearchResult
from .data_loader import IFSCodeDataset, DataAugmenter


@dataclass
class TrainingConfig:
    """
    Configuration for training the semantic search model.

    TRAINING PHILOSOPHY:
    -------------------
    Our training configuration balances several concerns:

    1. CONVERGENCE: Learning rates and schedules that achieve good convergence
    2. GENERALIZATION: Regularization and validation to avoid overfitting
    3. EFFICIENCY: Batch sizes and GPU utilization for reasonable training time
    4. STABILITY: Gradient clipping and loss weighting for stable training
    5. PRODUCTION: Model checkpointing and export for deployment
    """

    # Model architecture
    model_name: str = "microsoft/codebert-base"
    embedding_dim: int = 768
    hidden_dim: int = 256
    num_ifs_classes: int = 50
    freeze_backbone: bool = False

    # Training hyperparameters
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_steps: int = 1000
    gradient_clip_norm: float = 1.0

    # Loss weighting
    similarity_weight: float = 1.0
    classification_weight: float = 0.3
    regression_weight: float = 0.2
    auxiliary_weight: float = 0.1

    # Data settings
    max_sequence_length: int = 512
    negative_samples_per_positive: int = 3

    # Training control
    patience: int = 3  # Early stopping patience
    min_improvement: float = 0.001
    save_every_n_epochs: int = 1
    validate_every_n_steps: int = 500

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    mixed_precision: bool = True  # Use automatic mixed precision

    # Output paths
    output_dir: Path = Path("models/semantic_search")
    checkpoint_dir: Path = Path("checkpoints/semantic_search")
    log_dir: Path = Path("logs/semantic_search")

    def __post_init__(self):
        """Create directories and validate config."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Validate GPU settings
        if self.device == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA requested but not available, using CPU")
            self.device = "cpu"
            self.mixed_precision = False

        # Adjust batch size for available memory
        if self.device == "cuda":
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            if gpu_memory < 8 * 1024**3:  # Less than 8GB
                self.batch_size = min(self.batch_size, 8)
                logging.info(
                    f"Reduced batch size to {self.batch_size} for limited GPU memory"
                )


class SemanticSearchDataset(Dataset):
    """
    PyTorch Dataset for semantic search training.

    DATASET PHILOSOPHY:
    ------------------
    We create training samples that teach the model:

    1. POSITIVE PAIRS: Code chunks and queries that should match
    2. NEGATIVE PAIRS: Code chunks and queries that should NOT match
    3. MULTI-TASK TARGETS: Module, layer, quality labels for auxiliary tasks
    4. CONTRASTIVE TRIPLETS: Anchor, positive, negative for similarity learning

    The dataset dynamically generates varied training examples to improve
    generalization and prevent overfitting to specific patterns.
    """

    def __init__(
        self,
        chunks: List[CodeChunk],
        augmenter: DataAugmenter,
        tokenizer,
        config: TrainingConfig,
        mode: str = "train",
    ):
        """
        Initialize the dataset.

        Args:
            chunks: List of code chunks for training
            augmenter: Data augmenter for query generation
            tokenizer: Tokenizer for text processing
            config: Training configuration
            mode: 'train' or 'validation'
        """
        self.chunks = chunks
        self.augmenter = augmenter
        self.tokenizer = tokenizer
        self.config = config
        self.mode = mode

        # Create mappings for categorical labels
        self.module_to_id = self._create_module_mapping()
        self.layer_to_id = {
            "presentation": 0,
            "business": 1,
            "data": 2,
            "integration": 3,
        }

        # Pre-generate queries for efficiency (cache commonly used queries)
        self.chunk_queries = self._pregenerate_queries()

        # Create negative sampling pools for efficient sampling
        self.negative_pools = self._create_negative_pools()

        logging.info(f"Created {mode} dataset with {len(chunks)} chunks")
        logging.info(
            f"Generated {sum(len(queries) for queries in self.chunk_queries.values())} queries"
        )

    def __len__(self) -> int:
        """Dataset length based on number of chunks and query multiplier."""
        return len(self.chunks) * 2  # Average 2 queries per chunk

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.

        SAMPLE GENERATION STRATEGY:
        ---------------------------
        For each request, we:
        1. Select a target chunk (cycling through all chunks)
        2. Generate or select a positive query
        3. Sample negative chunks for contrastive learning
        4. Create auxiliary task targets
        5. Tokenize all text inputs
        6. Return formatted sample
        """
        # Determine target chunk (cycle through all chunks)
        chunk_idx = idx % len(self.chunks)
        target_chunk = self.chunks[chunk_idx]

        # Select a positive query for this chunk
        queries = self.chunk_queries.get(target_chunk.chunk_id, [])
        if queries:
            # Randomly select one of the pre-generated queries
            query_idx = np.random.randint(len(queries))
            positive_query = queries[query_idx]
        else:
            # Fallback to chunk's embedding text
            positive_query = target_chunk.to_embedding_text()

        # Sample negative chunks
        negative_chunks = self.augmenter.generate_negative_samples(
            target_chunk,
            self.chunks,
            num_negatives=self.config.negative_samples_per_positive,
        )

        # Tokenize inputs
        tokenized = self._tokenize_inputs(positive_query, target_chunk, negative_chunks)

        # Create auxiliary task targets
        targets = self._create_targets(target_chunk)

        # Combine into final sample
        sample = {**tokenized, **targets, "chunk_id": target_chunk.chunk_id}

        return sample

    def _create_module_mapping(self) -> Dict[str, int]:
        """Create mapping from module names to integer IDs."""
        modules = set()
        for chunk in self.chunks:
            if chunk.module:
                modules.add(chunk.module)

        # Sort for consistent mapping
        sorted_modules = sorted(modules)
        module_to_id = {module: i for i, module in enumerate(sorted_modules)}

        # Add unknown module
        module_to_id["UNKNOWN"] = len(module_to_id)

        logging.info(f"Created module mapping with {len(module_to_id)} modules")
        return module_to_id

    def _pregenerate_queries(self) -> Dict[str, List[str]]:
        """Pre-generate queries for all chunks to speed up training."""
        chunk_queries = {}

        logging.info("Pre-generating queries for training...")
        for chunk in tqdm(self.chunks, desc="Generating queries"):
            queries = self.augmenter.generate_queries_for_chunk(chunk)
            if queries:
                chunk_queries[chunk.chunk_id] = queries

        return chunk_queries

    def _create_negative_pools(self) -> Dict[str, List[CodeChunk]]:
        """Create pools of negative samples organized by similarity type."""
        pools = {
            "different_module": [],
            "different_layer": [],
            "similar_name": [],
            "random": self.chunks.copy(),
        }

        # Group chunks by module and layer for efficient negative sampling
        module_groups = {}
        layer_groups = {}

        for chunk in self.chunks:
            if chunk.module:
                if chunk.module not in module_groups:
                    module_groups[chunk.module] = []
                module_groups[chunk.module].append(chunk)

            if chunk.layer:
                if chunk.layer not in layer_groups:
                    layer_groups[chunk.layer] = []
                layer_groups[chunk.layer].append(chunk)

        # Store groups for efficient access
        self.module_groups = module_groups
        self.layer_groups = layer_groups

        return pools

    def _tokenize_inputs(
        self, query: str, positive_chunk: CodeChunk, negative_chunks: List[CodeChunk]
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize query and code chunks.

        TOKENIZATION STRATEGY:
        ---------------------
        We tokenize:
        1. Query text (user's search query)
        2. Positive chunk (code that should match)
        3. Negative chunks (code that should NOT match)

        All inputs are padded to same length and include attention masks.
        """
        if self.tokenizer:
            # Tokenize query
            query_tokens = self.tokenizer(
                query,
                padding="max_length",
                truncation=True,
                max_length=self.config.max_sequence_length,
                return_tensors="pt",
            )

            # Tokenize positive chunk
            positive_text = positive_chunk.to_embedding_text()
            positive_tokens = self.tokenizer(
                positive_text,
                padding="max_length",
                truncation=True,
                max_length=self.config.max_sequence_length,
                return_tensors="pt",
            )

            # Tokenize negative chunks
            negative_texts = [chunk.to_embedding_text() for chunk in negative_chunks]
            negative_tokens = self.tokenizer(
                negative_texts,
                padding="max_length",
                truncation=True,
                max_length=self.config.max_sequence_length,
                return_tensors="pt",
            )

            return {
                "query_input_ids": query_tokens["input_ids"].squeeze(0),
                "query_attention_mask": query_tokens["attention_mask"].squeeze(0),
                "positive_input_ids": positive_tokens["input_ids"].squeeze(0),
                "positive_attention_mask": positive_tokens["attention_mask"].squeeze(0),
                "negative_input_ids": negative_tokens["input_ids"],
                "negative_attention_mask": negative_tokens["attention_mask"],
            }
        else:
            # Fallback tokenization for when transformers isn't available
            return self._fallback_tokenize(query, positive_chunk, negative_chunks)

    def _fallback_tokenize(
        self, query: str, positive_chunk: CodeChunk, negative_chunks: List[CodeChunk]
    ) -> Dict[str, torch.Tensor]:
        """Simple tokenization fallback."""
        max_len = self.config.max_sequence_length

        def simple_tokenize(text: str) -> Tuple[torch.Tensor, torch.Tensor]:
            words = text.split()[:max_len]
            input_ids = torch.zeros(max_len, dtype=torch.long)
            attention_mask = torch.zeros(max_len, dtype=torch.long)

            for i, word in enumerate(words):
                input_ids[i] = abs(hash(word)) % 50000  # Simple vocab mapping
                attention_mask[i] = 1

            return input_ids, attention_mask

        # Tokenize all inputs
        query_ids, query_mask = simple_tokenize(query)
        pos_ids, pos_mask = simple_tokenize(positive_chunk.to_embedding_text())

        neg_ids_list = []
        neg_mask_list = []
        for chunk in negative_chunks:
            neg_ids, neg_mask = simple_tokenize(chunk.to_embedding_text())
            neg_ids_list.append(neg_ids)
            neg_mask_list.append(neg_mask)

        return {
            "query_input_ids": query_ids,
            "query_attention_mask": query_mask,
            "positive_input_ids": pos_ids,
            "positive_attention_mask": pos_mask,
            "negative_input_ids": torch.stack(neg_ids_list),
            "negative_attention_mask": torch.stack(neg_mask_list),
        }

    def _create_targets(self, chunk: CodeChunk) -> Dict[str, torch.Tensor]:
        """Create target labels for auxiliary tasks."""
        # Module classification target
        module_id = self.module_to_id.get(
            chunk.module or "UNKNOWN", self.module_to_id["UNKNOWN"]
        )

        # Layer classification target
        layer_id = self.layer_to_id.get(chunk.layer or "business", 1)

        # Quality regression target (normalize complexity score)
        quality_target = float(chunk.complexity_score)

        return {
            "module_label": torch.tensor(module_id, dtype=torch.long),
            "layer_label": torch.tensor(layer_id, dtype=torch.long),
            "quality_target": torch.tensor(quality_target, dtype=torch.float),
        }


class SemanticTrainer:
    """
    Trainer class for the IFS semantic search model.

    TRAINING PIPELINE PHILOSOPHY:
    ----------------------------
    Our training pipeline implements modern deep learning best practices:

    1. MIXED PRECISION: Faster training with automatic mixed precision
    2. GRADIENT ACCUMULATION: Simulate larger batch sizes with limited memory
    3. LEARNING RATE SCHEDULING: Warm-up and decay for stable convergence
    4. EARLY STOPPING: Prevent overfitting with validation monitoring
    5. CHECKPOINTING: Save and resume training from any point
    6. MONITORING: Comprehensive logging and visualization
    """

    def __init__(
        self,
        config: TrainingConfig,
        train_chunks: List[CodeChunk],
        val_chunks: Optional[List[CodeChunk]] = None,
    ):
        """
        Initialize the trainer.

        Args:
            config: Training configuration
            train_chunks: Training code chunks
            val_chunks: Validation code chunks (optional)
        """
        self.config = config
        self.train_chunks = train_chunks
        self.val_chunks = val_chunks or []

        # Setup logging
        self.setup_logging()

        # Initialize model
        self.model = IFSSemanticModel(
            model_name=config.model_name,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            num_ifs_classes=config.num_ifs_classes,
            freeze_backbone=config.freeze_backbone,
        ).to(config.device)

        logging.info(
            f"Model initialized with {count_parameters(self.model)[0]:,} parameters"
        )

        # Initialize loss function
        self.criterion = MultiTaskLoss(
            similarity_weight=config.similarity_weight,
            classification_weight=config.classification_weight,
            regression_weight=config.regression_weight,
            auxiliary_weight=config.auxiliary_weight,
        )

        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Initialize data components
        self.augmenter = DataAugmenter()
        self.setup_data_loaders()

        # Initialize scheduler
        self.scheduler = self.setup_scheduler()

        # Mixed precision training
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []

    def setup_logging(self):
        """Setup training logging."""
        log_file = (
            self.config.log_dir / f"training_{time.strftime('%Y%m%d_%H%M%S')}.log"
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

        logging.info("=" * 60)
        logging.info("STARTING IFS SEMANTIC SEARCH TRAINING")
        logging.info("=" * 60)
        logging.info(f"Configuration: {asdict(self.config)}")

    def setup_data_loaders(self):
        """Setup training and validation data loaders."""
        # Create datasets
        self.train_dataset = SemanticSearchDataset(
            chunks=self.train_chunks,
            augmenter=self.augmenter,
            tokenizer=self.model.tokenizer,
            config=self.config,
            mode="train",
        )

        if self.val_chunks:
            self.val_dataset = SemanticSearchDataset(
                chunks=self.val_chunks,
                augmenter=self.augmenter,
                tokenizer=self.model.tokenizer,
                config=self.config,
                mode="validation",
            )
        else:
            self.val_dataset = None

        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True if self.config.device == "cuda" else False,
        )

        if self.val_dataset:
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True if self.config.device == "cuda" else False,
            )
        else:
            self.val_loader = None

        logging.info(f"Training batches per epoch: {len(self.train_loader)}")
        if self.val_loader:
            logging.info(f"Validation batches: {len(self.val_loader)}")

    def setup_scheduler(self):
        """Setup learning rate scheduler."""
        total_steps = len(self.train_loader) * self.config.num_epochs

        if HAS_TRANSFORMERS:
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.warmup_steps,
                num_training_steps=total_steps,
            )
        else:
            # Fallback scheduler
            scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=total_steps,
                pct_start=self.config.warmup_steps / total_steps,
            )

        logging.info(
            f"Scheduler setup for {total_steps} total steps with {self.config.warmup_steps} warmup"
        )
        return scheduler

    def train(self) -> Dict[str, Any]:
        """
        Main training loop.

        TRAINING LOOP STRUCTURE:
        -----------------------
        For each epoch:
        1. Train on all batches
        2. Validate if validation data available
        3. Update learning rate schedule
        4. Save checkpoints
        5. Check early stopping
        6. Log progress and metrics

        Returns:
            Dictionary with training results and final model path
        """
        logging.info("Starting training...")

        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            logging.info(
                f"\n{'='*20} EPOCH {epoch + 1}/{self.config.num_epochs} {'='*20}"
            )

            # GPU memory monitoring
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                logging.info(
                    f"GPU Memory before epoch - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
                )

            # Training phase
            train_metrics = self.train_epoch()

            # Validation phase
            if self.val_loader:
                val_metrics = self.validate_epoch()
            else:
                val_metrics = {}

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()

            # Log epoch results
            self.log_epoch_results(train_metrics, val_metrics)

            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch, train_metrics, val_metrics)

            # GPU memory cleanup after each epoch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()  # Force garbage collection
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                logging.info(
                    f"GPU Memory after cleanup - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB"
                )

            # Early stopping check
            if self.val_loader and self.should_stop_early(val_metrics):
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Training complete
        total_time = time.time() - start_time
        logging.info(f"\nTraining completed in {total_time/3600:.2f} hours")

        # Save final model
        final_model_path = self.save_final_model()

        # Generate training report
        report = self.generate_training_report(total_time)

        return {
            "model_path": final_model_path,
            "training_time": total_time,
            "final_metrics": self.train_metrics[-1] if self.train_metrics else {},
            "report": report,
        }

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {
            "total_loss": 0.0,
            "similarity_loss": 0.0,
            "module_loss": 0.0,
            "layer_loss": 0.0,
            "quality_loss": 0.0,
            "batches": 0,
        }

        progress_bar = tqdm(
            self.train_loader, desc=f"Training Epoch {self.current_epoch + 1}"
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {
                k: v.to(self.config.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            # Forward pass
            loss_dict = self.forward_batch(batch)
            total_loss = loss_dict["total_loss"]

            # Backward pass
            if self.scaler:
                # Mixed precision training
                self.scaler.scale(total_loss).backward()
                if self.config.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_norm
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Regular training
                total_loss.backward()
                if self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.gradient_clip_norm
                    )
                self.optimizer.step()

            self.optimizer.zero_grad()

            # Update metrics
            for key, value in loss_dict.items():
                if key in epoch_metrics:
                    epoch_metrics[key] += value.item()
            epoch_metrics["batches"] += 1

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "Loss": f"{total_loss.item():.4f}",
                    "LR": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                }
            )

            self.global_step += 1

            # Validation during training
            if (
                self.val_loader
                and self.global_step % self.config.validate_every_n_steps == 0
            ):
                val_metrics = self.validate_epoch()
                logging.info(f"Step {self.global_step} validation: {val_metrics}")
                self.model.train()  # Return to training mode

        # Average metrics
        for key in epoch_metrics:
            if key != "batches" and epoch_metrics["batches"] > 0:
                epoch_metrics[key] /= epoch_metrics["batches"]

        return epoch_metrics

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        if not self.val_loader:
            return {}

        self.model.eval()
        val_metrics = {
            "total_loss": 0.0,
            "similarity_loss": 0.0,
            "module_loss": 0.0,
            "layer_loss": 0.0,
            "quality_loss": 0.0,
            "batches": 0,
        }

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move batch to device
                batch = {
                    k: v.to(self.config.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()
                }

                # Forward pass
                loss_dict = self.forward_batch(batch)

                # Update metrics
                for key, value in loss_dict.items():
                    if key in val_metrics:
                        val_metrics[key] += value.item()
                val_metrics["batches"] += 1

        # Average metrics
        for key in val_metrics:
            if key != "batches" and val_metrics["batches"] > 0:
                val_metrics[key] /= val_metrics["batches"]

        return val_metrics

    def forward_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass for a single batch."""
        # Get embeddings for query and positive/negative chunks
        query_outputs = self.model(
            batch["query_input_ids"], batch["query_attention_mask"]
        )

        positive_outputs = self.model(
            batch["positive_input_ids"], batch["positive_attention_mask"]
        )

        # Handle negative samples (multiple per batch item)
        batch_size = batch["negative_input_ids"].size(0)
        num_negatives = batch["negative_input_ids"].size(1)

        # Reshape negatives for batch processing
        neg_input_ids = batch["negative_input_ids"].view(
            -1, batch["negative_input_ids"].size(-1)
        )
        neg_attention_mask = batch["negative_attention_mask"].view(
            -1, batch["negative_attention_mask"].size(-1)
        )

        negative_outputs = self.model(neg_input_ids, neg_attention_mask)

        # Reshape negative embeddings back
        negative_embeddings = negative_outputs["embeddings"].view(
            batch_size, num_negatives, -1
        )

        # Prepare model outputs for loss computation
        model_outputs = {
            "anchor_embeddings": query_outputs["embeddings"],
            "positive_embeddings": positive_outputs["embeddings"],
            "negative_embeddings": negative_embeddings[:, 0, :],  # Use first negative
            "module_logits": positive_outputs["module_logits"],
            "layer_logits": positive_outputs["layer_logits"],
            "quality_score": positive_outputs["quality_score"],
        }

        # Prepare targets
        targets = {
            "module_labels": batch["module_label"],
            "layer_labels": batch["layer_label"],
            "quality_targets": batch["quality_target"],
        }

        # Compute losses
        loss_dict = self.criterion(model_outputs, targets)

        return loss_dict

    def should_stop_early(self, val_metrics: Dict[str, float]) -> bool:
        """Check if training should stop early based on validation metrics."""
        current_val_loss = val_metrics.get("total_loss", float("inf"))

        if current_val_loss < self.best_val_loss - self.config.min_improvement:
            self.best_val_loss = current_val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config.patience:
                return True
            return False

    def save_checkpoint(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "best_val_loss": self.best_val_loss,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": asdict(self.config),
        }

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        checkpoint_path = (
            self.config.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
        )
        torch.save(checkpoint, checkpoint_path)

        logging.info(f"Checkpoint saved: {checkpoint_path}")

    def save_final_model(self) -> Path:
        """Save final model for production deployment."""
        # Create metadata
        metadata = {
            "training_completed": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_epochs": self.current_epoch + 1,
            "total_steps": self.global_step,
            "best_val_loss": self.best_val_loss,
            "train_chunks_count": len(self.train_chunks),
            "val_chunks_count": len(self.val_chunks),
            "final_train_metrics": self.train_metrics[-1] if self.train_metrics else {},
            "final_val_metrics": self.val_metrics[-1] if self.val_metrics else {},
        }

        # Save model
        final_path = self.config.output_dir / "final_model"
        self.model.save_model(final_path, metadata)

        logging.info(f"Final model saved: {final_path}")
        return final_path

    def log_epoch_results(self, train_metrics: Dict, val_metrics: Dict):
        """Log results for an epoch."""
        self.train_metrics.append(train_metrics)
        if val_metrics:
            self.val_metrics.append(val_metrics)

        logging.info(f"Epoch {self.current_epoch + 1} Results:")
        logging.info(f"  Train Loss: {train_metrics.get('total_loss', 0):.4f}")
        if val_metrics:
            logging.info(f"  Val Loss:   {val_metrics.get('total_loss', 0):.4f}")
        logging.info(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")

    def generate_training_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive training report."""
        report = {
            "training_summary": {
                "total_epochs": self.current_epoch + 1,
                "total_steps": self.global_step,
                "total_time_hours": total_time / 3600,
                "final_learning_rate": self.optimizer.param_groups[0]["lr"],
                "best_validation_loss": self.best_val_loss,
            },
            "data_summary": {
                "training_chunks": len(self.train_chunks),
                "validation_chunks": len(self.val_chunks),
                "training_samples_per_epoch": len(self.train_dataset),
                "validation_samples": len(self.val_dataset) if self.val_dataset else 0,
            },
            "model_summary": {
                "total_parameters": count_parameters(self.model)[0],
                "trainable_parameters": count_parameters(self.model)[1],
                "embedding_dimension": self.config.embedding_dim,
                "backbone_model": self.config.model_name,
            },
            "final_metrics": {
                "train": self.train_metrics[-1] if self.train_metrics else {},
                "validation": self.val_metrics[-1] if self.val_metrics else {},
            },
        }

        # Save report
        report_path = self.config.log_dir / "training_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logging.info(f"Training report saved: {report_path}")
        return report


def create_train_val_split(
    chunks: List[CodeChunk], val_ratio: float = 0.2, stratify_by: str = "module"
) -> Tuple[List[CodeChunk], List[CodeChunk]]:
    """
    Create train/validation split with stratification.

    STRATIFICATION STRATEGY:
    -----------------------
    We stratify the split by module to ensure both train and validation
    sets have representative samples from all IFS modules. This prevents
    the model from overfitting to specific modules and ensures proper
    evaluation across the full domain.
    """
    if stratify_by == "module":
        # Group by module
        module_groups = {}
        for chunk in chunks:
            module = chunk.module or "UNKNOWN"
            if module not in module_groups:
                module_groups[module] = []
            module_groups[module].append(chunk)

        # Split each module group
        train_chunks = []
        val_chunks = []

        for module, module_chunks in module_groups.items():
            val_size = max(1, int(len(module_chunks) * val_ratio))

            # Shuffle within module
            np.random.shuffle(module_chunks)

            val_chunks.extend(module_chunks[:val_size])
            train_chunks.extend(module_chunks[val_size:])

        logging.info(
            f"Stratified split by module: {len(train_chunks)} train, {len(val_chunks)} val"
        )

    else:
        # Simple random split
        np.random.shuffle(chunks)
        val_size = int(len(chunks) * val_ratio)
        val_chunks = chunks[:val_size]
        train_chunks = chunks[val_size:]

        logging.info(f"Random split: {len(train_chunks)} train, {len(val_chunks)} val")

    return train_chunks, val_chunks


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example training pipeline
    print("Testing semantic search training pipeline...")

    # This would normally load real chunks from IFS codebase
    from .data_structures import CodeChunk

    # Create dummy chunks for testing
    dummy_chunks = []
    for i in range(100):
        chunk = CodeChunk(
            chunk_id=f"chunk_{i}",
            file_path=f"/path/to/file_{i}.plsql",
            start_line=1,
            end_line=50,
            raw_content=f"FUNCTION test_function_{i}() IS BEGIN ... END;",
            processed_content=f"test function {i} content",
            chunk_type="plsql_function",
            function_name=f"test_function_{i}",
            module=["ORDER", "INVOICE", "INVENTORY"][i % 3],
            layer=["business", "data", "integration"][i % 3],
            complexity_score=np.random.random(),
        )
        dummy_chunks.append(chunk)

    # Create train/val split
    train_chunks, val_chunks = create_train_val_split(dummy_chunks, val_ratio=0.2)

    # Create config
    config = TrainingConfig(
        num_epochs=2,  # Short test run
        batch_size=4,
        embedding_dim=128,  # Smaller for testing
        device="cpu",  # CPU for testing
    )

    # Create trainer
    trainer = SemanticTrainer(config, train_chunks, val_chunks)

    # Run training (would take longer with real data)
    print("Starting training test...")
    results = trainer.train()

    print(f"Training completed! Model saved to: {results['model_path']}")
    print(f"Training time: {results['training_time']:.2f} seconds")
