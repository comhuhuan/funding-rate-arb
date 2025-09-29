"""
Configuration management for the funding rate arbitrage system.

This module provides comprehensive configuration management with YAML loading,
environment variable resolution, validation, and hot-reload capabilities.
"""

import logging
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from decimal import Decimal

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class DatabaseConfig(BaseModel):
    """Database configuration."""
    host: str
    port: int = 5432
    name: str
    user: str
    password: str
    ssl_mode: Optional[str] = None
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600


class RedisConfig(BaseModel):
    """Redis configuration."""
    host: str
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False


class ExchangeConfig(BaseModel):
    """Exchange configuration."""
    timeout: int = 30000
    rateLimit: int = 1000
    enableRateLimit: bool = True
    sandbox: bool = False


class ArbitrageConfig(BaseModel):
    """Arbitrage configuration."""
    min_profit_threshold: Decimal = Field(default=Decimal("0.01"))
    max_position_size: Decimal = Field(default=Decimal("10000"))
    max_open_positions: int = 5
    max_daily_loss: Decimal = Field(default=Decimal("1000"))

    @field_validator('min_profit_threshold', 'max_position_size', 'max_daily_loss', mode='before')
    @classmethod
    def convert_to_decimal(cls, v):
        return Decimal(str(v)) if not isinstance(v, Decimal) else v


class RiskConfig(BaseModel):
    """Risk management configuration."""
    var_confidence: float = 0.95
    var_window: int = 100
    max_leverage: Decimal = Field(default=Decimal("3.0"))
    max_portfolio_risk: Decimal = Field(default=Decimal("0.02"))

    @field_validator('max_leverage', 'max_portfolio_risk', mode='before')
    @classmethod
    def convert_to_decimal(cls, v):
        return Decimal(str(v)) if not isinstance(v, Decimal) else v


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format_type: str = "standard"


class APIConfig(BaseModel):
    """API server configuration."""
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False


@dataclass
class SystemConfig:
    """Complete system configuration."""
    system: Dict[str, Any] = field(default_factory=dict)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    database: Optional[DatabaseConfig] = None
    redis: Optional[RedisConfig] = None
    exchanges: Dict[str, Any] = field(default_factory=dict)
    arbitrage: ArbitrageConfig = field(default_factory=ArbitrageConfig)
    risk_management: RiskConfig = field(default_factory=RiskConfig)
    api: APIConfig = field(default_factory=APIConfig)
    market_data: Dict[str, Any] = field(default_factory=dict)
    monitoring: Dict[str, Any] = field(default_factory=dict)


class ConfigManager:
    """
    Comprehensive configuration manager.

    Features:
    - YAML configuration file loading
    - Environment variable resolution with defaults
    - Configuration validation using Pydantic
    - Environment-specific overrides (development, production, etc.)
    - Hot-reload capabilities
    """

    ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}')

    def __init__(self, config_dir: str = "config", environment: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_dir: Directory containing configuration files
            environment: Environment name (development, production, etc.)
        """
        self.config_dir = Path(config_dir)
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self.config: Optional[SystemConfig] = None
        self._raw_config: Dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize the configuration manager."""
        try:
            self._load_configuration()
            self._validate_configuration()
            logger.info(f"Configuration manager initialized for environment: {self.environment}")
        except Exception as e:
            logger.error(f"Failed to initialize configuration: {e}")
            raise

    def _load_configuration(self) -> None:
        """Load configuration from YAML files."""
        # Load base configuration
        base_config = self._load_yaml_file("base.yaml")

        # Load environment-specific configuration
        env_config_file = f"{self.environment}.yaml"
        env_config = self._load_yaml_file(env_config_file)

        # Merge configurations (environment overrides base)
        self._raw_config = self._deep_merge(base_config, env_config)

        # Resolve environment variables
        self._raw_config = self._resolve_env_vars(self._raw_config)

    def _load_yaml_file(self, filename: str) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        file_path = self.config_dir / filename

        if not file_path.exists():
            if filename == "base.yaml":
                raise FileNotFoundError(f"Required base configuration file not found: {file_path}")
            logger.warning(f"Environment configuration file not found: {file_path}")
            return {}

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = yaml.safe_load(file) or {}
                logger.debug(f"Loaded configuration from {filename}")
                return content
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {filename}: {e}")
        except Exception as e:
            raise IOError(f"Failed to read {filename}: {e}")

    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    def _resolve_env_vars(self, config: Union[Dict, list, str, Any]) -> Any:
        """Recursively resolve environment variables in configuration."""
        if isinstance(config, dict):
            return {key: self._resolve_env_vars(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._resolve_env_vars(item) for item in config]
        elif isinstance(config, str):
            return self._substitute_env_vars(config)
        else:
            return config

    def _substitute_env_vars(self, value: str) -> str:
        """Substitute environment variables in a string."""
        def replace_var(match):
            var_expr = match.group(1)

            # Handle default values: ${VAR:default}
            if ':' in var_expr:
                var_name, default_value = var_expr.split(':', 1)
                return os.getenv(var_name, default_value)
            else:
                env_value = os.getenv(var_expr)
                if env_value is None:
                    raise ValueError(f"Required environment variable not set: {var_expr}")
                return env_value

        return self.ENV_VAR_PATTERN.sub(replace_var, value)

    def _validate_configuration(self) -> None:
        """Validate and convert configuration to typed objects."""
        try:
            # Extract and validate individual sections
            logging_config = LoggingConfig(**self._raw_config.get("logging", {}))

            arbitrage_config = ArbitrageConfig(**self._raw_config.get("arbitrage", {}))

            risk_config = RiskConfig(**self._raw_config.get("risk_management", {}))

            api_config = APIConfig(**self._raw_config.get("api", {}))

            # Optional database configuration
            database_config = None
            if "database" in self._raw_config:
                database_config = DatabaseConfig(**self._raw_config["database"])

            # Optional Redis configuration
            redis_config = None
            if "redis" in self._raw_config:
                redis_config = RedisConfig(**self._raw_config["redis"])

            # Create the complete system configuration
            self.config = SystemConfig(
                system=self._raw_config.get("system", {}),
                logging=logging_config,
                database=database_config,
                redis=redis_config,
                exchanges=self._raw_config.get("exchanges", {}),
                arbitrage=arbitrage_config,
                risk_management=risk_config,
                api=api_config,
                market_data=self._raw_config.get("market_data", {}),
                monitoring=self._raw_config.get("monitoring", {})
            )

        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}")

    def get_config(self) -> SystemConfig:
        """Get the current configuration."""
        if self.config is None:
            raise RuntimeError("Configuration not initialized. Call initialize() first.")
        return self.config

    def get_database_config(self) -> Optional[DatabaseConfig]:
        """Get database configuration."""
        return self.config.database if self.config else None

    def get_redis_config(self) -> Optional[RedisConfig]:
        """Get Redis configuration."""
        return self.config.redis if self.config else None

    def get_exchange_config(self, exchange_name: str) -> ExchangeConfig:
        """Get configuration for a specific exchange."""
        if not self.config:
            raise RuntimeError("Configuration not initialized")

        exchange_configs = self.config.exchanges.get("exchanges", {})
        default_config = self.config.exchanges.get("default_config", {})

        # Merge default config with exchange-specific config
        exchange_config = {**default_config, **exchange_configs.get(exchange_name, {})}

        return ExchangeConfig(**exchange_config)

    def reload_configuration(self) -> None:
        """Reload configuration from files."""
        logger.info("Reloading configuration...")
        try:
            self._load_configuration()
            self._validate_configuration()
            logger.info("Configuration reloaded successfully")
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            raise