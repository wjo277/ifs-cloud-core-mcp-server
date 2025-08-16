"""
Model Configuration for IFS Cloud MCP Server

This module provides configuration options for the intent classification model,
including the ability to switch between original and quantized models.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any

# Configuration file path
CONFIG_FILE = Path(__file__).parent / "config" / "model_config.json"

# Default configuration
DEFAULT_CONFIG = {
    "intent_classifier": {
        "use_quantized": True,  # Use quantized model by default for optimal performance
        "use_gpu": None,  # Auto-detect GPU availability
        "model_path": None,  # Use default path
    },
    "performance": {
        "target_inference_time_ms": 15,  # Target inference time (quantized is faster)
        "memory_optimization": True,  # Enable memory optimizations
    },
    "fallback": {
        "enable_keyword_fallback": True,  # Use keyword classification if ML fails
        "confidence_threshold": 0.7,  # Minimum confidence for ML prediction
    },
}


class ModelConfig:
    """Configuration manager for ML models."""

    def __init__(self):
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r") as f:
                    config = json.load(f)
                # Merge with defaults to ensure all keys exist
                return self._merge_config(DEFAULT_CONFIG, config)
            except Exception as e:
                print(f"Warning: Failed to load config file: {e}")
                print("Using default configuration")

        # Create config directory and file with defaults
        CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()

    def _merge_config(self, default: Dict, loaded: Dict) -> Dict:
        """Recursively merge loaded config with defaults."""
        result = default.copy()
        for key, value in loaded.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result

    def _save_config(self, config: Dict[str, Any]):
        """Save configuration to file."""
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save config file: {e}")

    def get_intent_classifier_config(self) -> Dict[str, Any]:
        """Get intent classifier configuration."""
        return self.config["intent_classifier"]

    def is_quantized_enabled(self) -> bool:
        """Check if quantized model should be used."""
        return self.config["intent_classifier"].get("use_quantized", True)

    def get_gpu_setting(self):
        """Get GPU usage setting."""
        return self.config["intent_classifier"].get("use_gpu")

    def get_model_path(self):
        """Get custom model path if specified."""
        return self.config["intent_classifier"].get("model_path")

    def set_quantized_enabled(self, enabled: bool):
        """Enable or disable quantized model."""
        self.config["intent_classifier"]["use_quantized"] = enabled
        self._save_config(self.config)

    def set_gpu_enabled(self, enabled: bool):
        """Enable or disable GPU usage."""
        self.config["intent_classifier"]["use_gpu"] = enabled
        self._save_config(self.config)

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration."""
        return self.config["performance"]

    def get_fallback_config(self) -> Dict[str, Any]:
        """Get fallback configuration."""
        return self.config["fallback"]

    def print_current_config(self):
        """Print current configuration in a readable format."""
        print("\n" + "=" * 60)
        print("🔧 CURRENT MODEL CONFIGURATION")
        print("=" * 60)

        intent_config = self.get_intent_classifier_config()
        perf_config = self.get_performance_config()
        fallback_config = self.get_fallback_config()

        print(f"\n🤖 Intent Classifier:")
        print(
            f"   Use Quantized Model: {intent_config['use_quantized']} {'✅' if intent_config['use_quantized'] else '❌'}"
        )
        print(
            f"   Use GPU: {intent_config['use_gpu']} {'(Auto-detect)' if intent_config['use_gpu'] is None else ''}"
        )
        print(f"   Custom Model Path: {intent_config['model_path'] or 'Default'}")

        print(f"\n⚡ Performance:")
        print(f"   Target Inference Time: {perf_config['target_inference_time_ms']} ms")
        print(
            f"   Memory Optimization: {perf_config['memory_optimization']} {'✅' if perf_config['memory_optimization'] else '❌'}"
        )

        print(f"\n🔄 Fallback:")
        print(
            f"   Keyword Fallback: {fallback_config['enable_keyword_fallback']} {'✅' if fallback_config['enable_keyword_fallback'] else '❌'}"
        )
        print(f"   Confidence Threshold: {fallback_config['confidence_threshold']}")

        print(f"\n📁 Config File: {CONFIG_FILE}")
        print("=" * 60 + "\n")


# Global configuration instance
model_config = ModelConfig()


def get_model_config() -> ModelConfig:
    """Get the global model configuration instance."""
    return model_config


def create_optimized_classifier():
    """Create a classifier with optimized settings."""
    from .proper_fastai_classifier import FastAIIntentClassifier

    config = get_model_config()
    intent_config = config.get_intent_classifier_config()

    return FastAIIntentClassifier(
        model_path=intent_config.get("model_path"),
        use_gpu=intent_config.get("use_gpu"),
        use_quantized=intent_config.get("use_quantized", True),
    )


if __name__ == "__main__":
    # Print current configuration
    config = get_model_config()
    config.print_current_config()

    # Example: Switch to original model
    # config.set_quantized_enabled(False)
    # print("Switched to original model")

    # Example: Enable GPU explicitly
    # config.set_gpu_enabled(True)
    # print("Enabled GPU usage")
