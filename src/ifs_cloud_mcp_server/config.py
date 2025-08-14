"""Configuration management for IFS Cloud MCP Server."""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict


logger = logging.getLogger(__name__)


@dataclass
class IFSCloudConfig:
    """Configuration for IFS Cloud MCP Server."""

    core_codes_path: Optional[str] = None
    last_indexed: Optional[str] = None
    index_statistics: Dict[str, Any] = None

    def __post_init__(self):
        if self.index_statistics is None:
            self.index_statistics = {}


class ConfigManager:
    """Manages configuration for the IFS Cloud MCP Server."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the configuration manager.

        Args:
            config_path: Path to the configuration file
        """
        if config_path is None:
            config_path = self._get_default_config_path()

        self.config_path = config_path
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config = self._load_config()

    def _get_default_config_path(self) -> Path:
        """Get the platform-appropriate default configuration path."""
        app_name = "ifs_cloud_mcp_server"

        if sys.platform == "win32":
            # Windows: %APPDATA%/ifs_cloud_mcp_server
            base_path = Path(
                os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming")
            )
        elif sys.platform == "darwin":
            # macOS: ~/Library/Application Support/ifs_cloud_mcp_server
            base_path = Path.home() / "Library" / "Application Support"
        else:
            # Linux/Unix: ~/.local/share/ifs_cloud_mcp_server
            base_path = Path(
                os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
            )

        return base_path / app_name / "config.json"

    def _load_config(self) -> IFSCloudConfig:
        """Load configuration from file."""
        if not self.config_path.exists():
            return IFSCloudConfig()

        try:
            with open(self.config_path, "r") as f:
                data = json.load(f)
            return IFSCloudConfig(**data)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to load config: {e}. Using defaults.")
            return IFSCloudConfig()

    def save_config(self) -> bool:
        """Save configuration to file.

        Returns:
            True if successful
        """
        try:
            with open(self.config_path, "w") as f:
                json.dump(asdict(self._config), f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False

    def get_core_codes_path(self) -> Optional[str]:
        """Get the configured IFS Cloud Core Codes path."""
        return self._config.core_codes_path

    def set_core_codes_path(self, path: str) -> bool:
        """Set the IFS Cloud Core Codes path.

        Args:
            path: Path to the IFS Cloud Core Codes directory

        Returns:
            True if path is valid and was set
        """
        path_obj = Path(path)
        if not path_obj.exists() or not path_obj.is_dir():
            return False

        self._config.core_codes_path = str(path_obj.absolute())
        return self.save_config()

    def get_last_indexed(self) -> Optional[str]:
        """Get the timestamp of the last indexing operation."""
        return self._config.last_indexed

    def set_last_indexed(self, timestamp: str) -> bool:
        """Set the timestamp of the last indexing operation."""
        self._config.last_indexed = timestamp
        return self.save_config()

    def get_index_statistics(self) -> Dict[str, Any]:
        """Get index statistics."""
        return self._config.index_statistics or {}

    def set_index_statistics(self, stats: Dict[str, Any]) -> bool:
        """Set index statistics."""
        self._config.index_statistics = stats
        return self.save_config()
