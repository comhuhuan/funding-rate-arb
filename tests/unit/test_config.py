"""
Unit tests for the configuration management system.
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path
from decimal import Decimal
from unittest.mock import patch

from src.infrastructure.config.manager import (
    ConfigManager,
    SystemConfig,
    DatabaseConfig,
    RedisConfig,
    ExchangeConfig,
    ArbitrageConfig,
    RiskConfig,
    LoggingConfig,
    APIConfig
)


class TestConfigModels:
    """Test configuration model classes."""

    def test_database_config_creation(self):
        """Test DatabaseConfig creation and validation."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            name="test_db",
            user="test_user",
            password="test_pass"
        )

        assert config.host == "localhost"
        assert config.port == 5432
        assert config.name == "test_db"
        assert config.user == "test_user"
        assert config.password == "test_pass"
        assert config.ssl_mode is None
        assert config.pool_size == 10

    def test_redis_config_creation(self):
        """Test RedisConfig creation and validation."""
        config = RedisConfig(
            host="localhost",
            port=6379,
            db=0
        )

        assert config.host == "localhost"
        assert config.port == 6379
        assert config.db == 0
        assert config.password is None
        assert config.ssl is False

    def test_exchange_config_creation(self):
        """Test ExchangeConfig creation and validation."""
        config = ExchangeConfig(
            timeout=30000,
            rateLimit=1000,
            enableRateLimit=True,
            sandbox=False
        )

        assert config.timeout == 30000
        assert config.rateLimit == 1000
        assert config.enableRateLimit is True
        assert config.sandbox is False

    def test_arbitrage_config_decimal_conversion(self):
        """Test ArbitrageConfig converts values to Decimal."""
        config = ArbitrageConfig(
            min_profit_threshold=0.01,
            max_position_size=10000.5,
            max_daily_loss="1000.0"
        )

        assert isinstance(config.min_profit_threshold, Decimal)
        assert config.min_profit_threshold == Decimal("0.01")
        assert isinstance(config.max_position_size, Decimal)
        assert config.max_position_size == Decimal("10000.5")
        assert isinstance(config.max_daily_loss, Decimal)
        assert config.max_daily_loss == Decimal("1000.0")

    def test_risk_config_decimal_conversion(self):
        """Test RiskConfig converts values to Decimal."""
        config = RiskConfig(
            max_leverage=2.5,
            max_portfolio_risk="0.05"
        )

        assert isinstance(config.max_leverage, Decimal)
        assert config.max_leverage == Decimal("2.5")
        assert isinstance(config.max_portfolio_risk, Decimal)
        assert config.max_portfolio_risk == Decimal("0.05")

    def test_system_config_defaults(self):
        """Test SystemConfig with default values."""
        config = SystemConfig()

        assert isinstance(config.system, dict)
        assert isinstance(config.logging, LoggingConfig)
        assert config.database is None
        assert config.redis is None
        assert isinstance(config.exchanges, dict)
        assert isinstance(config.arbitrage, ArbitrageConfig)
        assert isinstance(config.risk_management, RiskConfig)
        assert isinstance(config.api, APIConfig)


class TestConfigManager:
    """Test ConfigManager functionality."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for configuration files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_base_config(self):
        """Sample base configuration."""
        return {
            "system": {
                "name": "test-system",
                "version": "1.0.0",
                "timezone": "UTC"
            },
            "logging": {
                "level": "INFO",
                "format_type": "standard"
            },
            "arbitrage": {
                "min_profit_threshold": 0.01,
                "max_position_size": 10000
            },
            "risk_management": {
                "max_leverage": 3.0,
                "max_portfolio_risk": 0.02
            },
            "api": {
                "host": "127.0.0.1",
                "port": 8000
            }
        }

    @pytest.fixture
    def sample_dev_config(self):
        """Sample development configuration."""
        return {
            "environment": "development",
            "debug": True,
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "test_db",
                "user": "test_user",
                "password": "test_pass"
            },
            "redis": {
                "host": "localhost",
                "port": 6379,
                "db": 0
            }
        }

    def create_config_files(self, temp_dir, base_config, dev_config):
        """Helper to create configuration files."""
        import yaml

        base_file = Path(temp_dir) / "base.yaml"
        with open(base_file, 'w') as f:
            yaml.dump(base_config, f)

        dev_file = Path(temp_dir) / "development.yaml"
        with open(dev_file, 'w') as f:
            yaml.dump(dev_config, f)

    @pytest.mark.asyncio
    async def test_config_manager_initialization(self, temp_config_dir, sample_base_config, sample_dev_config):
        """Test ConfigManager initialization with YAML files."""
        self.create_config_files(temp_config_dir, sample_base_config, sample_dev_config)

        manager = ConfigManager(config_dir=temp_config_dir, environment="development")
        await manager.initialize()

        config = manager.get_config()

        assert config.system["name"] == "test-system"
        assert config.system["version"] == "1.0.0"
        assert config.logging.level == "INFO"
        assert config.database.host == "localhost"
        assert config.database.name == "test_db"
        assert config.redis.host == "localhost"

    @pytest.mark.asyncio
    async def test_config_manager_missing_base_file(self, temp_config_dir):
        """Test ConfigManager fails when base.yaml is missing."""
        manager = ConfigManager(config_dir=temp_config_dir)

        with pytest.raises(FileNotFoundError, match="Required base configuration file not found"):
            await manager.initialize()

    @pytest.mark.asyncio
    async def test_config_manager_missing_env_file(self, temp_config_dir, sample_base_config):
        """Test ConfigManager handles missing environment file gracefully."""
        import yaml

        base_file = Path(temp_config_dir) / "base.yaml"
        with open(base_file, 'w') as f:
            yaml.dump(sample_base_config, f)

        manager = ConfigManager(config_dir=temp_config_dir, environment="production")
        await manager.initialize()

        config = manager.get_config()
        assert config.system["name"] == "test-system"

    def test_environment_variable_resolution(self, temp_config_dir):
        """Test environment variable resolution."""
        import yaml

        config_with_env_vars = {
            "database": {
                "host": "${DB_HOST:localhost}",
                "port": "${DB_PORT:5432}",
                "name": "${DB_NAME}",
                "user": "${DB_USER:default_user}",
                "password": "${DB_PASSWORD:default_pass}"
            }
        }

        base_file = Path(temp_config_dir) / "base.yaml"
        with open(base_file, 'w') as f:
            yaml.dump(config_with_env_vars, f)

        with patch.dict(os.environ, {"DB_NAME": "production_db", "DB_HOST": "prod-server"}):
            manager = ConfigManager(config_dir=temp_config_dir)
            manager._load_configuration()

            assert manager._raw_config["database"]["host"] == "prod-server"
            assert manager._raw_config["database"]["port"] == "5432"  # default value
            assert manager._raw_config["database"]["name"] == "production_db"
            assert manager._raw_config["database"]["user"] == "default_user"  # default value

    def test_environment_variable_missing_required(self, temp_config_dir):
        """Test environment variable resolution fails for missing required vars."""
        import yaml

        config_with_env_vars = {
            "database": {
                "host": "${REQUIRED_HOST}",
                "name": "test"
            }
        }

        base_file = Path(temp_config_dir) / "base.yaml"
        with open(base_file, 'w') as f:
            yaml.dump(config_with_env_vars, f)

        manager = ConfigManager(config_dir=temp_config_dir)

        with pytest.raises(ValueError, match="Required environment variable not set: REQUIRED_HOST"):
            manager._load_configuration()

    def test_deep_merge_configuration(self):
        """Test deep merging of configuration dictionaries."""
        manager = ConfigManager()

        base = {
            "system": {"name": "base", "version": "1.0"},
            "database": {"host": "localhost", "port": 5432},
            "nested": {"level1": {"level2": {"value": "base"}}}
        }

        override = {
            "system": {"name": "override"},
            "database": {"host": "override-host"},
            "nested": {"level1": {"level2": {"value": "override", "new_key": "new"}}},
            "new_section": {"key": "value"}
        }

        result = manager._deep_merge(base, override)

        assert result["system"]["name"] == "override"
        assert result["system"]["version"] == "1.0"  # preserved from base
        assert result["database"]["host"] == "override-host"
        assert result["database"]["port"] == 5432  # preserved from base
        assert result["nested"]["level1"]["level2"]["value"] == "override"
        assert result["nested"]["level1"]["level2"]["new_key"] == "new"
        assert result["new_section"]["key"] == "value"

    @pytest.mark.asyncio
    async def test_get_exchange_config(self, temp_config_dir):
        """Test getting exchange-specific configuration."""
        import yaml

        config = {
            "exchanges": {
                "default_config": {
                    "timeout": 30000,
                    "rateLimit": 1000,
                    "sandbox": False
                },
                "exchanges": {
                    "binance": {
                        "timeout": 10000
                    }
                }
            }
        }

        base_file = Path(temp_config_dir) / "base.yaml"
        with open(base_file, 'w') as f:
            yaml.dump(config, f)

        manager = ConfigManager(config_dir=temp_config_dir)
        await manager.initialize()

        # Test default exchange config
        okx_config = manager.get_exchange_config("okx")
        assert okx_config.timeout == 30000
        assert okx_config.rateLimit == 1000
        assert okx_config.sandbox is False

        # Test exchange-specific config override
        binance_config = manager.get_exchange_config("binance")
        assert binance_config.timeout == 10000  # overridden
        assert binance_config.rateLimit == 1000  # from default
        assert binance_config.sandbox is False  # from default

    @pytest.mark.asyncio
    async def test_configuration_reload(self, temp_config_dir, sample_base_config, sample_dev_config):
        """Test configuration reload functionality."""
        import yaml

        self.create_config_files(temp_config_dir, sample_base_config, sample_dev_config)

        manager = ConfigManager(config_dir=temp_config_dir, environment="development")
        await manager.initialize()

        config = manager.get_config()
        assert config.system["name"] == "test-system"

        # Modify configuration file
        modified_base_config = sample_base_config.copy()
        modified_base_config["system"]["name"] = "modified-system"

        base_file = Path(temp_config_dir) / "base.yaml"
        with open(base_file, 'w') as f:
            yaml.dump(modified_base_config, f)

        # Reload configuration
        manager.reload_configuration()

        config = manager.get_config()
        assert config.system["name"] == "modified-system"

    def test_get_config_before_initialization(self):
        """Test getting config before initialization raises error."""
        manager = ConfigManager()

        with pytest.raises(RuntimeError, match="Configuration not initialized"):
            manager.get_config()

    def test_get_exchange_config_before_initialization(self):
        """Test getting exchange config before initialization raises error."""
        manager = ConfigManager()

        with pytest.raises(RuntimeError, match="Configuration not initialized"):
            manager.get_exchange_config("binance")

    @pytest.mark.asyncio
    async def test_invalid_yaml_file(self, temp_config_dir):
        """Test handling of invalid YAML file."""
        base_file = Path(temp_config_dir) / "base.yaml"
        with open(base_file, 'w') as f:
            f.write("invalid: yaml: content: [unclosed")

        manager = ConfigManager(config_dir=temp_config_dir)

        with pytest.raises(ValueError, match="Invalid YAML in base.yaml"):
            await manager.initialize()

    @pytest.mark.asyncio
    async def test_configuration_validation_error(self, temp_config_dir):
        """Test configuration validation error handling."""
        import yaml

        invalid_config = {
            "database": {
                "host": "localhost",
                "port": "invalid_port",  # Should be integer
                "name": "test",
                "user": "test",
                "password": "test"
            }
        }

        base_file = Path(temp_config_dir) / "base.yaml"
        with open(base_file, 'w') as f:
            yaml.dump(invalid_config, f)

        manager = ConfigManager(config_dir=temp_config_dir)

        with pytest.raises(ValueError, match="Configuration validation failed"):
            await manager.initialize()

    @pytest.mark.asyncio
    async def test_get_specific_configs(self, temp_config_dir, sample_base_config, sample_dev_config):
        """Test getting specific configuration sections."""
        self.create_config_files(temp_config_dir, sample_base_config, sample_dev_config)

        manager = ConfigManager(config_dir=temp_config_dir, environment="development")
        await manager.initialize()

        db_config = manager.get_database_config()
        assert db_config is not None
        assert db_config.host == "localhost"
        assert db_config.name == "test_db"

        redis_config = manager.get_redis_config()
        assert redis_config is not None
        assert redis_config.host == "localhost"
        assert redis_config.port == 6379

    @pytest.mark.asyncio
    async def test_optional_configs_none(self, temp_config_dir, sample_base_config):
        """Test that optional configs return None when not provided."""
        import yaml

        base_file = Path(temp_config_dir) / "base.yaml"
        with open(base_file, 'w') as f:
            yaml.dump(sample_base_config, f)

        manager = ConfigManager(config_dir=temp_config_dir)
        await manager.initialize()

        db_config = manager.get_database_config()
        assert db_config is None

        redis_config = manager.get_redis_config()
        assert redis_config is None