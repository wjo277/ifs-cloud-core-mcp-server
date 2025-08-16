"""
Intent Classification for IFS Cloud Search Queries

This module provides ML-based query intent classification to improve search ranking
by understanding what users are actually looking for.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


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
    """Result of intent classification."""

    intent: QueryIntent
    confidence: float
    probabilities: Dict[QueryIntent, float]


class IntentClassifier:
    """ML-based query intent classifier for IFS Cloud searches."""

    def __init__(self, model_path: Optional[Path] = None):
        self.vectorizer = TfidfVectorizer(
            max_features=1000, ngram_range=(1, 2), stop_words="english"
        )
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model_path = (
            model_path or Path(__file__).parent / "models" / "intent_classifier.pkl"
        )
        self.is_trained = False

        # Try to load existing model
        self._load_model()

    def generate_training_data(self) -> List[Tuple[str, QueryIntent]]:
        """Generate synthetic training data for IFS Cloud queries."""
        training_data = [
            # Business Logic queries
            ("expense authorization workflow", QueryIntent.BUSINESS_LOGIC),
            ("purchase order approval process", QueryIntent.BUSINESS_LOGIC),
            ("invoice validation rules", QueryIntent.BUSINESS_LOGIC),
            ("customer credit check procedure", QueryIntent.BUSINESS_LOGIC),
            ("payroll calculation logic", QueryIntent.BUSINESS_LOGIC),
            ("inventory reservation workflow", QueryIntent.BUSINESS_LOGIC),
            ("project cost calculation", QueryIntent.BUSINESS_LOGIC),
            ("employee status change process", QueryIntent.BUSINESS_LOGIC),
            ("supplier payment authorization", QueryIntent.BUSINESS_LOGIC),
            ("document approval workflow", QueryIntent.BUSINESS_LOGIC),
            ("budget control validation", QueryIntent.BUSINESS_LOGIC),
            ("time reporting business rules", QueryIntent.BUSINESS_LOGIC),
            ("quality control procedures", QueryIntent.BUSINESS_LOGIC),
            ("manufacturing order execution", QueryIntent.BUSINESS_LOGIC),
            ("financial posting logic", QueryIntent.BUSINESS_LOGIC),
            ("user access control", QueryIntent.BUSINESS_LOGIC),
            ("data validation rules", QueryIntent.BUSINESS_LOGIC),
            ("workflow automation", QueryIntent.BUSINESS_LOGIC),
            ("business rule engine", QueryIntent.BUSINESS_LOGIC),
            ("authorization check", QueryIntent.BUSINESS_LOGIC),
            # Entity Definition queries
            ("customer entity structure", QueryIntent.ENTITY_DEFINITION),
            ("employee data model", QueryIntent.ENTITY_DEFINITION),
            ("purchase order fields", QueryIntent.ENTITY_DEFINITION),
            ("invoice entity attributes", QueryIntent.ENTITY_DEFINITION),
            ("project entity definition", QueryIntent.ENTITY_DEFINITION),
            ("supplier table schema", QueryIntent.ENTITY_DEFINITION),
            ("inventory item properties", QueryIntent.ENTITY_DEFINITION),
            ("activity entity", QueryIntent.ENTITY_DEFINITION),
            ("customer order structure", QueryIntent.ENTITY_DEFINITION),
            ("person entity fields", QueryIntent.ENTITY_DEFINITION),
            ("company data structure", QueryIntent.ENTITY_DEFINITION),
            ("address entity definition", QueryIntent.ENTITY_DEFINITION),
            ("contact information fields", QueryIntent.ENTITY_DEFINITION),
            ("product entity attributes", QueryIntent.ENTITY_DEFINITION),
            ("order line structure", QueryIntent.ENTITY_DEFINITION),
            # UI Components queries
            ("customer order entry page", QueryIntent.UI_COMPONENTS),
            ("employee management form", QueryIntent.UI_COMPONENTS),
            ("purchase order navigator", QueryIntent.UI_COMPONENTS),
            ("invoice processing page", QueryIntent.UI_COMPONENTS),
            ("project management interface", QueryIntent.UI_COMPONENTS),
            ("inventory tracking screen", QueryIntent.UI_COMPONENTS),
            ("supplier registration form", QueryIntent.UI_COMPONENTS),
            ("report generation page", QueryIntent.UI_COMPONENTS),
            ("user profile settings", QueryIntent.UI_COMPONENTS),
            ("dashboard components", QueryIntent.UI_COMPONENTS),
            ("navigation menu", QueryIntent.UI_COMPONENTS),
            ("search interface", QueryIntent.UI_COMPONENTS),
            ("data entry form", QueryIntent.UI_COMPONENTS),
            ("list view", QueryIntent.UI_COMPONENTS),
            ("tree navigator", QueryIntent.UI_COMPONENTS),
            # API Integration queries
            ("customer order api", QueryIntent.API_INTEGRATION),
            ("employee service endpoint", QueryIntent.API_INTEGRATION),
            ("purchase order projection", QueryIntent.API_INTEGRATION),
            ("invoice api integration", QueryIntent.API_INTEGRATION),
            ("project management service", QueryIntent.API_INTEGRATION),
            ("inventory api calls", QueryIntent.API_INTEGRATION),
            ("supplier data service", QueryIntent.API_INTEGRATION),
            ("rest api endpoints", QueryIntent.API_INTEGRATION),
            ("web service integration", QueryIntent.API_INTEGRATION),
            ("api documentation", QueryIntent.API_INTEGRATION),
            ("service contract", QueryIntent.API_INTEGRATION),
            ("projection mapping", QueryIntent.API_INTEGRATION),
            ("external api", QueryIntent.API_INTEGRATION),
            ("integration points", QueryIntent.API_INTEGRATION),
            # Data Access queries
            ("customer order report", QueryIntent.DATA_ACCESS),
            ("employee status view", QueryIntent.DATA_ACCESS),
            ("purchase order history", QueryIntent.DATA_ACCESS),
            ("invoice summary report", QueryIntent.DATA_ACCESS),
            ("project cost view", QueryIntent.DATA_ACCESS),
            ("inventory levels report", QueryIntent.DATA_ACCESS),
            ("supplier performance view", QueryIntent.DATA_ACCESS),
            ("financial statements", QueryIntent.DATA_ACCESS),
            ("audit trail view", QueryIntent.DATA_ACCESS),
            ("management dashboard", QueryIntent.DATA_ACCESS),
            ("kpi reports", QueryIntent.DATA_ACCESS),
            ("data analytics view", QueryIntent.DATA_ACCESS),
            ("summary report", QueryIntent.DATA_ACCESS),
            ("detailed view", QueryIntent.DATA_ACCESS),
            # Troubleshooting queries
            ("error handling", QueryIntent.TROUBLESHOOTING),
            ("debug logging", QueryIntent.TROUBLESHOOTING),
            ("exception management", QueryIntent.TROUBLESHOOTING),
            ("error messages", QueryIntent.TROUBLESHOOTING),
            ("system diagnostics", QueryIntent.TROUBLESHOOTING),
            ("troubleshooting guide", QueryIntent.TROUBLESHOOTING),
            ("performance issues", QueryIntent.TROUBLESHOOTING),
            ("bug fixes", QueryIntent.TROUBLESHOOTING),
            ("system errors", QueryIntent.TROUBLESHOOTING),
            # General queries
            ("customer management", QueryIntent.GENERAL),
            ("employee information", QueryIntent.GENERAL),
            ("purchase orders", QueryIntent.GENERAL),
            ("invoicing", QueryIntent.GENERAL),
            ("project management", QueryIntent.GENERAL),
            ("inventory", QueryIntent.GENERAL),
            ("suppliers", QueryIntent.GENERAL),
            ("financial", QueryIntent.GENERAL),
            ("reporting", QueryIntent.GENERAL),
            ("user management", QueryIntent.GENERAL),
            ("system administration", QueryIntent.GENERAL),
            ("configuration", QueryIntent.GENERAL),
        ]

        return training_data

    def train(self, training_data: Optional[List[Tuple[str, QueryIntent]]] = None):
        """Train the intent classifier."""
        if training_data is None:
            training_data = self.generate_training_data()

        # Prepare data
        texts = [item[0] for item in training_data]
        labels = [item[1].value for item in training_data]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        # Vectorize
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        # Train classifier
        self.classifier.fit(X_train_vec, y_train)

        # Evaluate
        y_pred = self.classifier.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Intent Classifier Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        self.is_trained = True
        self._save_model()

    def predict(self, query: str) -> IntentPrediction:
        """Predict the intent of a query."""
        if not self.is_trained:
            # Return default for untrained model
            return IntentPrediction(
                intent=QueryIntent.GENERAL,
                confidence=0.5,
                probabilities={intent: 0.14 for intent in QueryIntent},
            )

        # Vectorize query
        query_vec = self.vectorizer.transform([query])

        # Get prediction and probabilities
        predicted_label = self.classifier.predict(query_vec)[0]
        probabilities = self.classifier.predict_proba(query_vec)[0]

        # Map probabilities to intents
        intent_probs = {}
        for i, intent in enumerate(QueryIntent):
            if intent.value in self.classifier.classes_:
                class_idx = list(self.classifier.classes_).index(intent.value)
                intent_probs[intent] = probabilities[class_idx]
            else:
                intent_probs[intent] = 0.0

        predicted_intent = QueryIntent(predicted_label)
        confidence = intent_probs[predicted_intent]

        return IntentPrediction(
            intent=predicted_intent, confidence=confidence, probabilities=intent_probs
        )

    def _save_model(self):
        """Save the trained model to disk."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "is_trained": self.is_trained,
        }

        with open(self.model_path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {self.model_path}")

    def _load_model(self):
        """Load a trained model from disk."""
        if self.model_path.exists():
            try:
                with open(self.model_path, "rb") as f:
                    model_data = pickle.load(f)

                self.vectorizer = model_data["vectorizer"]
                self.classifier = model_data["classifier"]
                self.is_trained = model_data["is_trained"]

                print(f"Model loaded from {self.model_path}")
            except Exception as e:
                print(f"Failed to load model: {e}")
                self.is_trained = False


def train_intent_classifier():
    """Utility function to train and save the intent classifier."""
    classifier = IntentClassifier()
    classifier.train()
    return classifier


if __name__ == "__main__":
    # Train the classifier
    classifier = train_intent_classifier()

    # Test some queries
    test_queries = [
        "expense authorization workflow",
        "customer entity structure",
        "purchase order entry page",
        "invoice api integration",
        "employee status report",
        "error handling",
    ]

    print("\n" + "=" * 60)
    print("TESTING INTENT CLASSIFIER")
    print("=" * 60)

    for query in test_queries:
        prediction = classifier.predict(query)
        print(f"\nQuery: '{query}'")
        print(
            f"Intent: {prediction.intent.value} (confidence: {prediction.confidence:.3f})"
        )

        # Show top 3 probabilities
        top_intents = sorted(
            prediction.probabilities.items(), key=lambda x: x[1], reverse=True
        )[:3]
        print("Top predictions:")
        for intent, prob in top_intents:
            print(f"  {intent.value}: {prob:.3f}")
