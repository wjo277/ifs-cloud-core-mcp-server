#!/usr/bin/env python3
"""
Proper FastAI Training Script for Intent Classifier

This script uses the actual FastAI framework with ULMFiT for high-level
text classification as requested.
"""

import os
import sys
from pathlib import Path


def check_fastai_setup():
    """Check if FastAI is properly set up."""
    try:
        import torch
        import fastai
        import fastai.text.all

        print("🔍 FastAI Setup Check")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"FastAI Version: {fastai.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA Version: {torch.version.cuda}")
            print(f"GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        return True

    except ImportError as e:
        print(f"❌ FastAI import error: {e}")
        return False


def train_proper_fastai_model():
    """Train the proper FastAI intent classifier using ULMFiT."""
    from ifs_cloud_mcp_server.proper_fastai_classifier import FastAIIntentClassifier

    print("\n" + "=" * 60)
    print("🚀 PROPER FASTAI TRAINING - ULMFiT Intent Classifier")
    print("=" * 60)

    # Initialize FastAI classifier
    classifier = FastAIIntentClassifier()

    # Check if already trained
    if classifier.is_trained:
        print("✅ Pre-trained FastAI ULMFiT model found!")

        # Test predictions
        test_queries = [
            "customer order workflow process",
            "inventory part database schema",
            "purchase order entry screen",
            "invoice processing api endpoint",
            "sales report generation",
            "database connection timeout error",
            "project architecture overview",
        ]

        print("\n🧪 Testing FastAI ULMFiT predictions:")
        for query in test_queries:
            pred = classifier.predict(query)
            print(f"'{query}' -> {pred.intent.value} ({pred.confidence:.3f})")

    else:
        print("🎯 Starting FastAI ULMFiT training...")
        print("📚 Using comprehensive IFS Cloud training data")
        print("🧠 FastAI will use transfer learning with ULMFiT")
        print(
            "⚙️ Features: automatic LR finding, discriminative fine-tuning, one-cycle training"
        )

        try:
            # Train the FastAI model
            classifier.train()
            print("\n✅ FastAI ULMFiT training completed!")

            # Test the trained model
            print("\n🧪 Testing trained FastAI model:")
            test_queries = [
                "customer authorization workflow",
                "database schema definition",
                "form validation interface",
                "rest api service",
                "inventory report view",
                "connection error debug",
                "system overview",
            ]

            for query in test_queries:
                pred = classifier.predict(query)
                print(f"'{query}' -> {pred.intent.value} ({pred.confidence:.3f})")

        except Exception as e:
            print(f"❌ FastAI training failed: {e}")
            import traceback

            traceback.print_exc()
            return False

    print("\n🎉 FastAI ULMFiT Advantages:")
    print("   - Transfer learning from pre-trained language model")
    print("   - Automatic learning rate finding")
    print("   - Discriminative fine-tuning")
    print("   - One-cycle training policy")
    print("   - Built-in regularization and dropout")
    print("   - Higher-level API with less boilerplate")
    print("   - Better generalization with less data")

    return True


if __name__ == "__main__":
    if check_fastai_setup():
        train_proper_fastai_model()
    else:
        print("❌ FastAI setup check failed")
        sys.exit(1)
