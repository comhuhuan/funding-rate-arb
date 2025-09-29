"""
Configuration management for the funding rate arbitrage system.

This module provides a minimal ConfigManager implementation that will be
expanded in the next development phase.
"""

import logging
from typing import Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SystemConfig:
    """Minimal system configuration."""
    system: Dict[str, Any]


class ConfigManager:
    """
    Minimal configuration manager.

    This is a placeholder implementation that will be expanded
    with full YAML loading, environment variable resolution,
    and hot-reload capabilities.
    """

    def __init__(self, config_dir: str = "config"):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = config_dir
        self.config: SystemConfig = None

    async def initialize(self) -> None:
        """Initialize the configuration manager."""
        # TODO: Implement full configuration loading
        # For now, use hardcoded minimal config
        self.config = SystemConfig(
            system={
                "name": "funding-rate-arb",
                "version": "1.0.0",
                "environment": "development",
                "debug": True,
                "timezone": "UTC"
            }
        )
        logger.info("Configuration manager initialized")

    def get_config(self) -> SystemConfig:
        """Get the current configuration."""
        return self.config