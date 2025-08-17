"""
Neural Network Models for IFS Semantic Search
=============================================

This module implements the neural architecture for semantic code search.
We use a FastAI-based approach that combines pre-trained language models
with IFS-specific adaptations for enterprise code understanding.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
import json
import logging

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    from transformers.modeling_outputs import BaseModelOutput

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logging.warning(
        "Transformers library not available. Using fallback implementation."
    )

try:
    import fastai
    from fastai.text.all import *
    from fastai.basics import *

    HAS_FASTAI = True
except ImportError:
    HAS_FASTAI = False
    logging.warning("FastAI library not available. Using PyTorch-only implementation.")


class IFSSemanticModel(nn.Module):
    """
    Main semantic search model for IFS code.

    ARCHITECTURE PHILOSOPHY:
    -----------------------
    Our model needs to understand both:
    1. GENERAL CODE PATTERNS: Leveraging pre-trained code models (CodeBERT, etc.)
    2. IFS-SPECIFIC PATTERNS: Business terminology, API usage, architectural patterns

    We achieve this through:
    - Pre-trained transformer backbone for general code understanding
    - IFS-specific adapter layers for domain knowledge
    - Multi-task learning for different types of searches
    - Contrastive learning for similarity ranking

    PRODUCTION CONSIDERATIONS:
    -------------------------
    - Model must be serializable for CPU inference
    - Embedding dimension chosen for FAISS efficiency
    - Memory footprint optimized for production deployment
    - GPU training → CPU inference pipeline supported
    """

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        embedding_dim: int = 768,
        hidden_dim: int = 256,
        num_ifs_classes: int = 50,  # Number of IFS modules/categories
        dropout_rate: float = 0.1,
        freeze_backbone: bool = False,
    ):
        """
        Initialize the IFS semantic model.

        Args:
            model_name: Pre-trained model name (CodeBERT, etc.)
            embedding_dim: Output embedding dimension
            hidden_dim: Hidden layer dimension
            num_ifs_classes: Number of IFS-specific categories for classification
            dropout_rate: Dropout rate for regularization
            freeze_backbone: Whether to freeze pre-trained weights
        """
        super().__init__()

        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Load pre-trained backbone
        if HAS_TRANSFORMERS:
            self.backbone = self._load_pretrained_backbone(model_name, freeze_backbone)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            backbone_dim = self.backbone.config.hidden_size
        else:
            # Fallback to simple LSTM for environments without transformers
            self.backbone = self._create_fallback_backbone()
            self.tokenizer = None
            backbone_dim = 256

        # IFS-specific adaptation layers
        self.ifs_adapter = IFSSpecificAdapter(
            input_dim=backbone_dim,
            hidden_dim=hidden_dim,
            num_classes=num_ifs_classes,
            dropout_rate=dropout_rate,
        )

        # Final embedding projection
        self.embedding_projection = nn.Sequential(
            nn.Linear(backbone_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # Multi-task heads for different types of predictions
        self.module_classifier = nn.Linear(embedding_dim, num_ifs_classes)
        self.layer_classifier = nn.Linear(
            embedding_dim, 4
        )  # presentation/business/data/integration
        self.quality_regressor = nn.Linear(embedding_dim, 1)  # Complexity/quality score

        # Store config for serialization
        self.config = {
            "model_name": model_name,
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "num_ifs_classes": num_ifs_classes,
            "dropout_rate": dropout_rate,
            "freeze_backbone": freeze_backbone,
        }

    def _load_pretrained_backbone(self, model_name: str, freeze: bool) -> nn.Module:
        """
        Load and configure pre-trained transformer backbone.

        BACKBONE SELECTION RATIONALE:
        ----------------------------
        - CodeBERT: Pre-trained on code, understands programming patterns
        - GraphCodeBERT: Includes data flow understanding
        - CodeT5: Good for code generation and understanding
        - RoBERTa: Strong general language understanding

        We default to CodeBERT as it's specifically designed for code.
        """
        try:
            model = AutoModel.from_pretrained(model_name)

            if freeze:
                # Freeze backbone weights for faster fine-tuning
                for param in model.parameters():
                    param.requires_grad = False
                logging.info(f"Frozen backbone model {model_name}")
            else:
                logging.info(f"Loaded trainable backbone model {model_name}")

            return model

        except Exception as e:
            logging.error(f"Failed to load {model_name}: {e}")
            logging.info("Falling back to simple backbone")
            return self._create_fallback_backbone()

    def _create_fallback_backbone(self) -> nn.Module:
        """
        Create a simple LSTM backbone when transformers aren't available.

        FALLBACK ARCHITECTURE:
        ---------------------
        When we can't use transformers (due to dependencies or resources),
        we fall back to a simpler architecture that can still capture
        some semantic relationships:
        - Character/word level embedding
        - Bidirectional LSTM
        - Attention pooling
        """
        return nn.ModuleDict(
            {
                "embedding": nn.Embedding(50000, 128),  # Large vocab for code
                "lstm": nn.LSTM(
                    128, 128, num_layers=2, bidirectional=True, batch_first=True
                ),
                "attention": nn.MultiheadAttention(256, num_heads=8, batch_first=True),
                "pooler": nn.Linear(256, 256),
            }
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        metadata: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        FORWARD PASS STRATEGY:
        ---------------------
        1. Pass text through backbone (transformer or LSTM)
        2. Extract pooled representation
        3. Apply IFS-specific adaptation
        4. Generate final embeddings and auxiliary predictions
        5. Return rich output for multiple loss functions

        Args:
            input_ids: Tokenized input text
            attention_mask: Attention mask for padding
            metadata: Additional metadata features (API calls, modules, etc.)

        Returns:
            Dictionary containing:
            - embeddings: Main similarity embeddings
            - module_logits: IFS module classification
            - layer_logits: Architectural layer classification
            - quality_score: Code quality/complexity score
        """
        batch_size = input_ids.size(0)

        # Backbone forward pass
        if HAS_TRANSFORMERS and hasattr(self.backbone, "config"):
            # Transformer backbone
            backbone_outputs = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )

            # Use pooler output if available, otherwise mean pool
            if (
                hasattr(backbone_outputs, "pooler_output")
                and backbone_outputs.pooler_output is not None
            ):
                backbone_features = backbone_outputs.pooler_output
            else:
                # Mean pooling over sequence dimension
                hidden_states = backbone_outputs.last_hidden_state
                if attention_mask is not None:
                    # Masked mean pooling
                    mask_expanded = attention_mask.unsqueeze(-1).expand(
                        hidden_states.size()
                    )
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    backbone_features = sum_embeddings / sum_mask
                else:
                    backbone_features = hidden_states.mean(dim=1)

        else:
            # Fallback LSTM backbone
            backbone_features = self._forward_fallback_backbone(
                input_ids, attention_mask
            )

        # IFS-specific adaptation
        ifs_features, ifs_predictions = self.ifs_adapter(backbone_features, metadata)

        # Combine backbone and IFS features
        combined_features = torch.cat([backbone_features, ifs_features], dim=-1)

        # Generate final embeddings
        embeddings = self.embedding_projection(combined_features)

        # Multi-task predictions
        module_logits = self.module_classifier(embeddings)
        layer_logits = self.layer_classifier(embeddings)
        quality_score = self.quality_regressor(embeddings).squeeze(-1)

        return {
            "embeddings": embeddings,
            "module_logits": module_logits,
            "layer_logits": layer_logits,
            "quality_score": quality_score,
            "ifs_predictions": ifs_predictions,
        }

    def _forward_fallback_backbone(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through fallback LSTM backbone."""
        # Simple character-level processing for fallback
        # In a real implementation, you'd want proper tokenization
        embeds = self.backbone["embedding"](input_ids)

        # LSTM processing
        lstm_out, (hidden, cell) = self.backbone["lstm"](embeds)

        # Attention pooling
        attn_out, attn_weights = self.backbone["attention"](
            lstm_out, lstm_out, lstm_out
        )

        # Mean pooling with attention to mask
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(attn_out.size())
            masked_out = attn_out * mask_expanded
            features = masked_out.sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            features = attn_out.mean(dim=1)

        return self.backbone["pooler"](features)

    def encode_text(self, texts: List[str], device: str = "cpu") -> np.ndarray:
        """
        Encode texts into embeddings for similarity search.

        ENCODING FOR PRODUCTION:
        -----------------------
        This method is used during production inference:
        1. Tokenize input texts
        2. Generate embeddings
        3. Normalize for cosine similarity
        4. Convert to numpy for FAISS indexing

        Optimized for:
        - Batch processing
        - Memory efficiency
        - CPU inference
        """
        self.eval()
        embeddings = []

        with torch.no_grad():
            # Process in batches to manage memory
            batch_size = 32
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                # Tokenize
                if self.tokenizer:
                    inputs = self.tokenizer(
                        batch_texts,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    )
                    input_ids = inputs["input_ids"].to(device)
                    attention_mask = inputs["attention_mask"].to(device)
                else:
                    # Fallback tokenization (very simple)
                    max_len = min(512, max(len(text.split()) for text in batch_texts))
                    input_ids = torch.zeros(
                        (len(batch_texts), max_len), dtype=torch.long
                    )
                    attention_mask = torch.ones_like(input_ids)

                    for j, text in enumerate(batch_texts):
                        words = text.split()[:max_len]
                        for k, word in enumerate(words):
                            # Simple hash-based vocab mapping
                            input_ids[j, k] = abs(hash(word)) % 50000

                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)

                # Forward pass
                outputs = self.forward(input_ids, attention_mask)
                batch_embeddings = outputs["embeddings"]

                # Normalize for cosine similarity
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

                embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)

    def save_model(self, save_path: Path, metadata: Optional[Dict] = None):
        """
        Save model for production deployment.

        PRODUCTION SERIALIZATION:
        ------------------------
        We save:
        1. Model weights (state_dict)
        2. Model configuration
        3. Training metadata
        4. Version information

        This allows loading on CPU for inference even if trained on GPU.
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model weights
        torch.save(self.state_dict(), save_path / "model_weights.pth")

        # Save configuration
        config_with_metadata = {
            **self.config,
            "model_type": "IFSSemanticModel",
            "version": "1.0",
            "embedding_dim": self.embedding_dim,
        }

        if metadata:
            config_with_metadata["training_metadata"] = metadata

        with open(save_path / "config.json", "w") as f:
            json.dump(config_with_metadata, f, indent=2)

        # Save tokenizer if available
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_path / "tokenizer")

        logging.info(f"Model saved to {save_path}")

    @classmethod
    def load_model(cls, load_path: Path, device: str = "cpu") -> "IFSSemanticModel":
        """
        Load model for production inference.

        PRODUCTION LOADING:
        ------------------
        Loads model weights and configuration from saved files.
        Automatically handles CPU/GPU device mapping.
        """
        load_path = Path(load_path)

        # Load configuration
        with open(load_path / "config.json", "r") as f:
            config = json.load(f)

        # Create model instance
        model = cls(
            model_name=config["model_name"],
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            num_ifs_classes=config["num_ifs_classes"],
            dropout_rate=config["dropout_rate"],
            freeze_backbone=config["freeze_backbone"],
        )

        # Load weights
        state_dict = torch.load(load_path / "model_weights.pth", map_location=device)
        model.load_state_dict(state_dict)

        # Load tokenizer if available
        tokenizer_path = load_path / "tokenizer"
        if tokenizer_path.exists() and HAS_TRANSFORMERS:
            model.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        model.to(device)
        model.eval()

        logging.info(f"Model loaded from {load_path} to {device}")
        return model


class IFSSpecificAdapter(nn.Module):
    """
    IFS-specific adaptation layer for domain knowledge.

    DOMAIN ADAPTATION PHILOSOPHY:
    ----------------------------
    While pre-trained models understand general code patterns,
    they don't understand IFS-specific patterns:

    1. BUSINESS TERMINOLOGY: Orders, Invoices, Inventory, etc.
    2. API PATTERNS: IFS_API calls, specific naming conventions
    3. ARCHITECTURAL PATTERNS: Entity/Projection/Fragment relationships
    4. MODULE RELATIONSHIPS: How different IFS modules interact

    This adapter learns these domain-specific patterns.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()

        # IFS business terminology embeddings
        self.business_term_embeddings = nn.Embedding(1000, 64)  # Top 1000 IFS terms
        self.api_embeddings = nn.Embedding(500, 32)  # IFS API patterns

        # Metadata processing layers
        self.metadata_processor = nn.Sequential(
            nn.Linear(input_dim + 64 + 32, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # IFS-specific predictions
        self.business_context_classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        backbone_features: torch.Tensor,
        metadata: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Process IFS-specific features and make domain predictions.

        Args:
            backbone_features: Features from the main backbone
            metadata: IFS-specific metadata (business terms, API calls, etc.)

        Returns:
            Tuple of (adapted_features, ifs_predictions)
        """
        batch_size = backbone_features.size(0)

        # Initialize metadata features with zeros if not provided
        if metadata is None:
            business_features = torch.zeros(
                batch_size, 64, device=backbone_features.device
            )
            api_features = torch.zeros(batch_size, 32, device=backbone_features.device)
        else:
            # Process business term metadata
            if "business_term_ids" in metadata:
                business_features = self.business_term_embeddings(
                    metadata["business_term_ids"]
                ).mean(dim=1)
            else:
                business_features = torch.zeros(
                    batch_size, 64, device=backbone_features.device
                )

            # Process API call metadata
            if "api_call_ids" in metadata:
                api_features = self.api_embeddings(metadata["api_call_ids"]).mean(dim=1)
            else:
                api_features = torch.zeros(
                    batch_size, 32, device=backbone_features.device
                )

        # Combine all features
        combined_features = torch.cat(
            [backbone_features, business_features, api_features], dim=-1
        )

        # Process through adaptation layers
        adapted_features = self.metadata_processor(combined_features)

        # Generate IFS-specific predictions
        business_context_logits = self.business_context_classifier(adapted_features)

        predictions = {"business_context_logits": business_context_logits}

        return adapted_features, predictions


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning semantic similarity.

    CONTRASTIVE LEARNING RATIONALE:
    ------------------------------
    For semantic search, we need embeddings where:
    1. Similar code has similar embeddings (small distance)
    2. Dissimilar code has dissimilar embeddings (large distance)
    3. The similarity aligns with human judgment of relevance

    Contrastive learning achieves this by:
    - Pulling positive pairs together
    - Pushing negative pairs apart
    - Learning from relative comparisons rather than absolute labels
    """

    def __init__(self, margin: float = 1.0, temperature: float = 0.05):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(
        self,
        anchor_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss between anchor, positive, and negative samples.

        LOSS COMPUTATION:
        ----------------
        We use InfoNCE-style contrastive loss:
        1. Compute similarities between anchor and all samples
        2. Apply temperature scaling
        3. Use cross-entropy to maximize positive similarity
        4. This naturally minimizes negative similarities
        """
        batch_size = anchor_embeddings.size(0)

        # Normalize embeddings for cosine similarity
        anchor_norm = F.normalize(anchor_embeddings, p=2, dim=1)
        positive_norm = F.normalize(positive_embeddings, p=2, dim=1)
        negative_norm = F.normalize(negative_embeddings, p=2, dim=1)

        # Compute similarities
        pos_sim = torch.sum(anchor_norm * positive_norm, dim=1) / self.temperature
        neg_sim = torch.sum(anchor_norm * negative_norm, dim=1) / self.temperature

        # InfoNCE loss
        logits = torch.stack([pos_sim, neg_sim], dim=1)
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)

        return loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining different training objectives.

    MULTI-TASK TRAINING RATIONALE:
    -----------------------------
    We train on multiple related tasks simultaneously:
    1. SIMILARITY: Main contrastive loss for search relevance
    2. CLASSIFICATION: Module/layer classification for structured understanding
    3. REGRESSION: Quality scoring for result ranking
    4. AUXILIARY: Business context understanding

    This multi-task approach:
    - Improves generalization through shared representations
    - Provides richer supervision signals
    - Creates more robust embeddings
    - Enables auxiliary capabilities (filtering, ranking)
    """

    def __init__(
        self,
        similarity_weight: float = 1.0,
        classification_weight: float = 0.3,
        regression_weight: float = 0.2,
        auxiliary_weight: float = 0.1,
    ):
        super().__init__()

        self.similarity_weight = similarity_weight
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        self.auxiliary_weight = auxiliary_weight

        self.contrastive_loss = ContrastiveLoss()
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()

    def forward(
        self, model_outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute weighted multi-task loss.

        Args:
            model_outputs: Dictionary of model predictions
            targets: Dictionary of ground truth targets

        Returns:
            Dictionary containing individual losses and total loss
        """
        losses = {}

        # Contrastive similarity loss
        if all(
            key in model_outputs
            for key in [
                "anchor_embeddings",
                "positive_embeddings",
                "negative_embeddings",
            ]
        ):
            similarity_loss = self.contrastive_loss(
                model_outputs["anchor_embeddings"],
                model_outputs["positive_embeddings"],
                model_outputs["negative_embeddings"],
            )
            losses["similarity_loss"] = similarity_loss * self.similarity_weight

        # Module classification loss
        if "module_logits" in model_outputs and "module_labels" in targets:
            module_loss = self.classification_loss(
                model_outputs["module_logits"], targets["module_labels"]
            )
            losses["module_loss"] = module_loss * self.classification_weight

        # Layer classification loss
        if "layer_logits" in model_outputs and "layer_labels" in targets:
            layer_loss = self.classification_loss(
                model_outputs["layer_logits"], targets["layer_labels"]
            )
            losses["layer_loss"] = layer_loss * self.classification_weight

        # Quality regression loss
        if "quality_score" in model_outputs and "quality_targets" in targets:
            quality_loss = self.regression_loss(
                model_outputs["quality_score"], targets["quality_targets"]
            )
            losses["quality_loss"] = quality_loss * self.regression_weight

        # Auxiliary business context loss
        if (
            "ifs_predictions" in model_outputs
            and "business_context_logits" in model_outputs["ifs_predictions"]
        ):
            if "business_context_labels" in targets:
                aux_loss = self.classification_loss(
                    model_outputs["ifs_predictions"]["business_context_logits"],
                    targets["business_context_labels"],
                )
                losses["auxiliary_loss"] = aux_loss * self.auxiliary_weight

        # Total loss
        total_loss = sum(losses.values())
        losses["total_loss"] = total_loss

        return losses


# Utility functions for model management
def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def freeze_layers(model: nn.Module, layer_names: List[str]):
    """Freeze specific layers in the model."""
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = False
            logging.info(f"Frozen layer: {name}")


def unfreeze_layers(model: nn.Module, layer_names: List[str]):
    """Unfreeze specific layers in the model."""
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):
            param.requires_grad = True
            logging.info(f"Unfrozen layer: {name}")


# Example usage and testing
if __name__ == "__main__":
    # Test model creation
    print("Testing IFS Semantic Model...")

    # Create model (will use fallback if transformers not available)
    model = IFSSemanticModel(
        model_name="microsoft/codebert-base" if HAS_TRANSFORMERS else "fallback",
        embedding_dim=256,
        hidden_dim=128,
    )

    print(f"Model created with {count_parameters(model)[0]:,} total parameters")

    # Test forward pass
    batch_size = 4
    seq_len = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)

    outputs = model(input_ids, attention_mask)

    print("Forward pass successful!")
    print(f"Embedding shape: {outputs['embeddings'].shape}")
    print(f"Module logits shape: {outputs['module_logits'].shape}")

    # Test encoding
    test_texts = [
        "function create_order() { ... }",
        "procedure update_invoice() is begin ... end;",
        "const processPayment = () => { ... }",
        "SELECT * FROM order_tab WHERE ...",
    ]

    embeddings = model.encode_text(test_texts)
    print(f"Generated embeddings shape: {embeddings.shape}")

    # Test similarity
    similarities = np.dot(embeddings, embeddings.T)
    print(f"Similarity matrix:\n{similarities}")

    print("All tests passed!")
