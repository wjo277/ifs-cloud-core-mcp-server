"""
Proper FastAI-based Intent Classification for IFS Cloud Search Queries

This module uses FastAI's ULMFiT approach for high-level text classification
with better results and less complexity than manual transformers setup.
"""

import json
import logging
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# FastAI imports
from fastai.text.all import *
import warnings

# Import model downloader
from .model_downloader import ensure_model_available, get_model_path

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Different types of query intents."""

    BUSINESS_LOGIC = "business_logic"  # authorization, validation, workflows
    ENTITY_DEFINITION = "entity_definition"  # data structure, schema
    UI_COMPONENTS = "ui_components"  # pages, forms, navigation
    API_INTEGRATION = "api_integration"  # projections, services
    DATA_ACCESS = "data_access"  # views, reports
    TROUBLESHOOTING = "troubleshooting"  # errors, debugging
    GENERAL = "general"  # broad topics


@dataclass
class IntentPrediction:
    """Prediction result with confidence."""

    intent: QueryIntent
    confidence: float
    all_scores: Dict[str, float]


class FastAIIntentClassifier:
    """Proper FastAI-based intent classifier using ULMFiT."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_gpu: Optional[bool] = None,
        use_quantized: bool = True,  # Default to quantized for better performance
    ):
        """Initialize the FastAI classifier."""
        # Device configuration
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()

        self.use_gpu = use_gpu
        self.use_quantized = use_quantized
        logger.info(f"Using GPU: {use_gpu}")
        logger.info(f"Using quantized model: {use_quantized}")

        # Model paths
        self.model_dir = (
            Path(model_path)
            if model_path
            else Path(__file__).parent / "models" / "fastai_intent"
        )
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Intent mapping
        self.intent_labels = [intent.value for intent in QueryIntent]

        # Model components
        self.learner = None
        self.dls = None
        self.is_trained = False

        # Try to load existing model
        self._load_model()

    def _load_model(self):
        """Load existing FastAI model if available, downloading from release if needed."""
        # Choose model file based on quantization preference
        if self.use_quantized:
            model_file = self.model_dir / "export_quantized.pkl"
            fallback_file = self.model_dir / "export.pkl"
        else:
            model_file = self.model_dir / "export.pkl"
            fallback_file = None

        # Try quantized model first if requested, fallback to original
        if not model_file.exists() and fallback_file and fallback_file.exists():
            logger.info(f"Quantized model not found, falling back to original model")
            model_file = fallback_file

        if not model_file.exists():
            # Try to download from GitHub releases - prefer quantized
            logger.info(
                "Model not found locally, attempting to download from releases..."
            )
            if ensure_model_available(use_quantized=self.use_quantized):
                # The downloader puts the model in the expected location
                if model_file.exists():
                    logger.info(f"Successfully downloaded model to {model_file}")
                elif fallback_file and fallback_file.exists():
                    logger.info(f"Downloaded fallback model to {fallback_file}")
                    model_file = fallback_file
                else:
                    logger.warning(
                        "Download appeared successful but model file not found"
                    )
                    return
            else:
                logger.warning("Failed to download model from releases")
                return

        if model_file.exists():
            try:
                # Set PyTorch loading to be less strict for FastAI models
                torch.serialization.clear_safe_globals()
                torch.serialization.add_safe_globals(
                    [
                        "fastcore.foundation.L",
                        "fastai.text.models.awd_lstm.AWD_LSTM",
                        "collections.OrderedDict",
                        "torch.nn.modules.container.Sequential",
                        "torch.nn.modules.linear.Linear",
                        "torch.nn.modules.dropout.Dropout",
                        "torch.nn.modules.activation.ReLU",
                        "fastai.text.models.core.LinearDecoder",
                        "fastai.torch_core.Module",
                    ]
                )

                self.learner = load_learner(model_file)
                self.is_trained = True
                model_type = (
                    "quantized" if "quantized" in str(model_file) else "original"
                )
                model_size = model_file.stat().st_size / (1024 * 1024)
                logger.info(
                    f"✅ Loaded FastAI {model_type} model successfully ({model_size:.1f} MB)"
                )
            except Exception as e:
                logger.warning(f"Failed to load FastAI model: {e}")
                self.is_trained = False

    def generate_training_data(self) -> List[Tuple[str, str]]:
        """Generate comprehensive training data for FastAI."""
        training_data = [
            # BUSINESS_LOGIC - Enhanced with more examples
            ("user authorization check", "business_logic"),
            ("validate customer order workflow", "business_logic"),
            ("approval process rules", "business_logic"),
            ("business rule validation", "business_logic"),
            ("workflow state transitions", "business_logic"),
            ("permission checks for invoice", "business_logic"),
            ("authorization matrix setup", "business_logic"),
            ("validation rules configuration", "business_logic"),
            ("business process flow", "business_logic"),
            ("state machine transitions", "business_logic"),
            ("role based access control", "business_logic"),
            ("conditional logic implementation", "business_logic"),
            ("trigger execution order", "business_logic"),
            ("event handling mechanism", "business_logic"),
            ("business constraint validation", "business_logic"),
            ("workflow approval hierarchy", "business_logic"),
            ("process automation rules", "business_logic"),
            ("business logic implementation", "business_logic"),
            ("validation framework setup", "business_logic"),
            ("authorization service configuration", "business_logic"),
            ("user permissions management", "business_logic"),
            ("business rule engine", "business_logic"),
            ("workflow orchestration", "business_logic"),
            ("process validation logic", "business_logic"),
            ("authentication mechanisms", "business_logic"),
            # ENTITY_DEFINITION - Enhanced with more examples
            ("customer entity definition", "entity_definition"),
            ("database table structure", "entity_definition"),
            ("entity relationship mapping", "entity_definition"),
            ("data model schema", "entity_definition"),
            ("column definitions and types", "entity_definition"),
            ("primary key constraints", "entity_definition"),
            ("foreign key relationships", "entity_definition"),
            ("entity attributes overview", "entity_definition"),
            ("table join specifications", "entity_definition"),
            ("data dictionary entries", "entity_definition"),
            ("object type definitions", "entity_definition"),
            ("record structure layout", "entity_definition"),
            ("field mappings and types", "entity_definition"),
            ("database schema design", "entity_definition"),
            ("entity class hierarchy", "entity_definition"),
            ("data type specifications", "entity_definition"),
            ("attribute constraints definition", "entity_definition"),
            ("relational model structure", "entity_definition"),
            ("entity metadata configuration", "entity_definition"),
            ("schema evolution tracking", "entity_definition"),
            ("table relationships diagram", "entity_definition"),
            ("database normalization rules", "entity_definition"),
            ("entity integrity constraints", "entity_definition"),
            ("data model documentation", "entity_definition"),
            ("schema migration scripts", "entity_definition"),
            # UI_COMPONENTS - Enhanced with more examples
            ("purchase order entry form", "ui_components"),
            ("customer details page layout", "ui_components"),
            ("navigation menu structure", "ui_components"),
            ("form field validation", "ui_components"),
            ("page component hierarchy", "ui_components"),
            ("user interface elements", "ui_components"),
            ("screen flow design", "ui_components"),
            ("form submission handling", "ui_components"),
            ("page routing configuration", "ui_components"),
            ("UI component library", "ui_components"),
            ("responsive layout design", "ui_components"),
            ("form builder configuration", "ui_components"),
            ("page template structure", "ui_components"),
            ("navigation breadcrumb setup", "ui_components"),
            ("modal dialog implementation", "ui_components"),
            ("tab panel organization", "ui_components"),
            ("grid component configuration", "ui_components"),
            ("search form interface", "ui_components"),
            ("dashboard widget layout", "ui_components"),
            ("menu item permissions", "ui_components"),
            ("user interface design", "ui_components"),
            ("form validation rules", "ui_components"),
            ("page navigation flow", "ui_components"),
            ("component styling setup", "ui_components"),
            ("interactive element configuration", "ui_components"),
            # API_INTEGRATION - Enhanced with more examples
            ("customer projection service", "api_integration"),
            ("REST API endpoint configuration", "api_integration"),
            ("service integration patterns", "api_integration"),
            ("API response mapping", "api_integration"),
            ("projection query optimization", "api_integration"),
            ("service orchestration layer", "api_integration"),
            ("API authentication setup", "api_integration"),
            ("external service integration", "api_integration"),
            ("microservice communication", "api_integration"),
            ("API gateway configuration", "api_integration"),
            ("service contract definition", "api_integration"),
            ("REST endpoint security", "api_integration"),
            ("API versioning strategy", "api_integration"),
            ("service discovery mechanism", "api_integration"),
            ("API rate limiting setup", "api_integration"),
            ("projection service performance", "api_integration"),
            ("service mesh configuration", "api_integration"),
            ("API monitoring and logging", "api_integration"),
            ("service dependency management", "api_integration"),
            ("API documentation generation", "api_integration"),
            ("web service implementation", "api_integration"),
            ("service endpoint routing", "api_integration"),
            ("API request handling", "api_integration"),
            ("service layer architecture", "api_integration"),
            ("integration middleware setup", "api_integration"),
            # DATA_ACCESS - Enhanced with more examples
            ("inventory status report", "data_access"),
            ("customer payment history view", "data_access"),
            ("sales analytics dashboard", "data_access"),
            ("financial report generation", "data_access"),
            ("data warehouse queries", "data_access"),
            ("business intelligence views", "data_access"),
            ("reporting framework setup", "data_access"),
            ("query performance optimization", "data_access"),
            ("data visualization components", "data_access"),
            ("report template design", "data_access"),
            ("analytical data processing", "data_access"),
            ("report scheduling system", "data_access"),
            ("data export functionality", "data_access"),
            ("query result caching", "data_access"),
            ("report parameter handling", "data_access"),
            ("data aggregation queries", "data_access"),
            ("report distribution setup", "data_access"),
            ("data access permissions", "data_access"),
            ("query execution planning", "data_access"),
            ("report formatting options", "data_access"),
            ("database view creation", "data_access"),
            ("data mining queries", "data_access"),
            ("report automation tools", "data_access"),
            ("data analysis reports", "data_access"),
            ("business metrics tracking", "data_access"),
            # TROUBLESHOOTING - Enhanced with more examples
            ("database connection error", "troubleshooting"),
            ("service timeout debugging", "troubleshooting"),
            ("performance issue analysis", "troubleshooting"),
            ("error message interpretation", "troubleshooting"),
            ("system health monitoring", "troubleshooting"),
            ("debugging workflow issues", "troubleshooting"),
            ("error logging configuration", "troubleshooting"),
            ("performance bottleneck identification", "troubleshooting"),
            ("system diagnostic tools", "troubleshooting"),
            ("error handling strategies", "troubleshooting"),
            ("troubleshooting guide creation", "troubleshooting"),
            ("system monitoring setup", "troubleshooting"),
            ("error pattern analysis", "troubleshooting"),
            ("debugging best practices", "troubleshooting"),
            ("system health checks", "troubleshooting"),
            ("error recovery procedures", "troubleshooting"),
            ("performance tuning guidelines", "troubleshooting"),
            ("diagnostic data collection", "troubleshooting"),
            ("issue escalation process", "troubleshooting"),
            ("system maintenance procedures", "troubleshooting"),
            ("application crash debugging", "troubleshooting"),
            ("memory leak investigation", "troubleshooting"),
            ("network connectivity issues", "troubleshooting"),
            ("security vulnerability fixes", "troubleshooting"),
            ("system optimization techniques", "troubleshooting"),
            # GENERAL - Enhanced with more examples
            ("project management overview", "general"),
            ("system architecture design", "general"),
            ("best practices documentation", "general"),
            ("implementation guidelines", "general"),
            ("technology stack overview", "general"),
            ("project planning strategies", "general"),
            ("development methodology", "general"),
            ("system integration approach", "general"),
            ("architectural patterns usage", "general"),
            ("project delivery framework", "general"),
            ("technical documentation standards", "general"),
            ("system design principles", "general"),
            ("development process optimization", "general"),
            ("technology evaluation criteria", "general"),
            ("project management tools", "general"),
            ("system architecture evolution", "general"),
            ("development best practices", "general"),
            ("project success metrics", "general"),
            ("technical decision making", "general"),
            ("system scalability planning", "general"),
            ("software development lifecycle", "general"),
            ("quality assurance processes", "general"),
            ("technical architecture review", "general"),
            ("system deployment strategies", "general"),
            ("project milestone planning", "general"),
        ]

        # Add variations for better training diversity
        extended_data = []
        for text, intent in training_data:
            extended_data.append((text, intent))

            # Add variations with synonyms
            if "customer" in text:
                extended_data.append((text.replace("customer", "client"), intent))
            if "order" in text:
                extended_data.append(
                    (text.replace("order", "purchase request"), intent)
                )
            if "configuration" in text:
                extended_data.append((text.replace("configuration", "setup"), intent))
            if "validation" in text:
                extended_data.append(
                    (text.replace("validation", "verification"), intent)
                )
            if "implementation" in text:
                extended_data.append(
                    (text.replace("implementation", "development"), intent)
                )

        return extended_data

    def train(self):
        """Train the FastAI model using ULMFiT approach."""
        training_data = self.generate_training_data()
        logger.info(f"🚀 Training FastAI model on {len(training_data)} examples")

        # Convert to DataFrame - FastAI's preferred format
        df = pd.DataFrame(training_data, columns=["text", "label"])
        logger.info(f"📊 Created DataFrame with {len(df)} rows")

        # Create DataLoaders - this is where FastAI shines
        self.dls = TextDataLoaders.from_df(
            df,
            text_col="text",
            label_col="label",
            valid_pct=0.2,  # 20% for validation
            seed=42,
            bs=32 if self.use_gpu else 16,  # Batch size
            seq_len=128,  # Sequence length
            device=torch.device("cuda" if self.use_gpu else "cpu"),
        )

        logger.info("✅ Created FastAI DataLoaders")

        # Create text classifier learner with ULMFiT
        self.learner = text_classifier_learner(
            self.dls,
            AWD_LSTM,  # The ULMFiT architecture
            drop_mult=0.5,  # Dropout multiplier
            metrics=accuracy,
        )

        logger.info("✅ Created FastAI text classifier with ULMFiT")

        # Find optimal learning rate - FastAI magic!
        logger.info("🔍 Finding optimal learning rate...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                lr_min, lr_steep = self.learner.lr_find(suggest_funcs=(minimum, steep))
                suggested_lr = lr_steep
                logger.info(f"📈 Suggested learning rate: {suggested_lr}")
            except Exception as e:
                logger.warning(f"LR finder failed: {e}, using default")
                suggested_lr = 1e-2

        # Train the head first (transfer learning approach) - using 1 epoch for testing
        logger.info("🎯 Training classifier head (frozen backbone) - 1 epoch test...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.learner.fit_one_cycle(1, suggested_lr)

        # Unfreeze and fine-tune entire model
        logger.info("🔓 Unfreezing model for fine-tuning...")
        self.learner.unfreeze()

        # Fine-tune with discriminative learning rates - using 1 epoch for testing
        logger.info("🎨 Fine-tuning entire model - 1 epoch test...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.learner.fit_one_cycle(1, slice(suggested_lr / 100, suggested_lr / 10))

        # Export the model - FastAI way
        export_path = self.model_dir / "export.pkl"
        self.learner.export(export_path)
        self.is_trained = True

        logger.info(f"✅ Training completed! FastAI model exported to {export_path}")

        # Save metadata
        metadata = {
            "num_examples": len(training_data),
            "num_labels": len(self.intent_labels),
            "intent_labels": self.intent_labels,
            "architecture": "ULMFiT (AWD_LSTM)",
            "framework": "FastAI",
        }

        with open(self.model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def predict(self, query: str) -> IntentPrediction:
        """Predict intent for a query using FastAI."""
        if not self.is_trained:
            logger.warning("Model not trained, falling back to general intent")
            return IntentPrediction(
                intent=QueryIntent.GENERAL, confidence=0.5, all_scores={"general": 0.5}
            )

        try:
            # FastAI prediction - simple and clean!
            pred_class, pred_idx, probs = self.learner.predict(query)

            # Convert to our format
            intent_name = str(pred_class)
            intent = QueryIntent(intent_name)
            confidence = float(probs.max())

            # Get all scores
            all_scores = {}
            for i, prob in enumerate(probs):
                intent_key = self.intent_labels[i]
                all_scores[intent_key] = float(prob)

            return IntentPrediction(
                intent=intent, confidence=confidence, all_scores=all_scores
            )

        except Exception as e:
            logger.error(f"FastAI prediction failed: {e}")
            return IntentPrediction(
                intent=QueryIntent.GENERAL, confidence=0.5, all_scores={"general": 0.5}
            )

    def evaluate(self, test_data: List[Tuple[str, str]]) -> Dict[str, float]:
        """Evaluate FastAI model performance."""
        if not self.is_trained:
            return {"error": "Model not trained"}

        correct = 0
        total = len(test_data)

        for query, expected_intent in test_data:
            prediction = self.predict(query)
            if prediction.intent.value == expected_intent:
                correct += 1

        accuracy = correct / total
        return {"accuracy": accuracy, "correct": correct, "total": total}
