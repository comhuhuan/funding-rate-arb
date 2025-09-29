"""
Redis cache manager for the funding rate arbitrage system.

This module provides Redis-based caching functionality with connection pooling,
serialization, compression, and error handling for high-performance trading systems.
"""

import asyncio
import json
import logging
import pickle
import zlib
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic
from decimal import Decimal

import redis.asyncio as redis
from redis.asyncio import ConnectionPool

from src.infrastructure.config.manager import RedisConfig
from src.types import FundingRate, Order, Symbol, ArbitrageOpportunity

logger = logging.getLogger(__name__)

T = TypeVar('T')


class SerializationError(Exception):
    """Exception raised when serialization/deserialization fails."""
    pass


class CacheError(Exception):
    """Exception raised when cache operations fail."""
    pass


class CacheSerializer(ABC):
    """Abstract base class for cache serialization."""

    @abstractmethod
    def serialize(self, obj: Any) -> bytes:
        """Serialize an object to bytes."""
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to an object."""
        pass


class JSONSerializer(CacheSerializer):
    """JSON-based serialization with Decimal support."""

    class DecimalEncoder(json.JSONEncoder):
        """JSON encoder that handles Decimal types."""
        def default(self, obj):
            if isinstance(obj, Decimal):
                return {'__decimal__': str(obj)}
            # Convert other non-serializable objects to strings as fallback
            return str(obj)

    @staticmethod
    def decimal_hook(dct):
        """JSON decoder hook for Decimal types."""
        if '__decimal__' in dct:
            return Decimal(dct['__decimal__'])
        return dct

    def serialize(self, obj: Any) -> bytes:
        """Serialize object to JSON bytes."""
        try:
            json_str = json.dumps(obj, cls=self.DecimalEncoder)
            return json_str.encode('utf-8')
        except (TypeError, ValueError) as e:
            raise SerializationError(f"Failed to serialize to JSON: {e}")

    def deserialize(self, data: bytes) -> Any:
        """Deserialize JSON bytes to object."""
        try:
            json_str = data.decode('utf-8')
            return json.loads(json_str, object_hook=self.decimal_hook)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise SerializationError(f"Failed to deserialize from JSON: {e}")


class PickleSerializer(CacheSerializer):
    """Pickle-based serialization for complex objects."""

    def serialize(self, obj: Any) -> bytes:
        """Serialize object using pickle."""
        try:
            return pickle.dumps(obj)
        except (TypeError, AttributeError) as e:
            raise SerializationError(f"Failed to serialize with pickle: {e}")

    def deserialize(self, data: bytes) -> Any:
        """Deserialize pickle bytes to object."""
        try:
            return pickle.loads(data)
        except (pickle.UnpicklingError, EOFError) as e:
            raise SerializationError(f"Failed to deserialize with pickle: {e}")


class CacheManager(Generic[T]):
    """
    Generic cache manager with compression and TTL support.

    This class provides a high-level interface for caching with:
    - Automatic serialization/deserialization
    - Optional compression for large objects
    - TTL (time-to-live) support
    - Type hints for better IDE support
    """

    def __init__(
        self,
        redis_client: redis.Redis,
        serializer: CacheSerializer,
        key_prefix: str = "",
        default_ttl: Optional[int] = None,
        compress_threshold: int = 1024,
        compression_level: int = 6
    ):
        """
        Initialize cache manager.

        Args:
            redis_client: Redis client instance
            serializer: Serialization strategy
            key_prefix: Prefix for all cache keys
            default_ttl: Default TTL in seconds
            compress_threshold: Compress data larger than this size (bytes)
            compression_level: Compression level (1-9)
        """
        self.redis = redis_client
        self.serializer = serializer
        self.key_prefix = key_prefix
        self.default_ttl = default_ttl
        self.compress_threshold = compress_threshold
        self.compression_level = compression_level

    def _make_key(self, key: str) -> str:
        """Create a full cache key with prefix."""
        return f"{self.key_prefix}:{key}" if self.key_prefix else key

    def _compress_data(self, data: bytes) -> tuple[bytes, bool]:
        """Compress data if it exceeds threshold."""
        if len(data) > self.compress_threshold:
            compressed = zlib.compress(data, self.compression_level)
            return compressed, True
        return data, False

    def _decompress_data(self, data: bytes, is_compressed: bool) -> bytes:
        """Decompress data if it was compressed."""
        if is_compressed:
            return zlib.decompress(data)
        return data

    async def set(
        self,
        key: str,
        value: T,
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False
    ) -> bool:
        """
        Set a cache value.

        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (uses default if None)
            nx: Only set if key doesn't exist
            xx: Only set if key exists

        Returns:
            True if set was successful
        """
        try:
            # Serialize the value
            serialized_data = self.serializer.serialize(value)

            # Compress if needed
            final_data, is_compressed = self._compress_data(serialized_data)

            # Add compression metadata
            if is_compressed:
                final_data = b'\x01' + final_data  # Compression marker
            else:
                final_data = b'\x00' + final_data  # No compression marker

            cache_key = self._make_key(key)
            ttl_value = ttl or self.default_ttl

            result = await self.redis.set(
                cache_key,
                final_data,
                ex=ttl_value,
                nx=nx,
                xx=xx
            )

            logger.debug(f"Set cache key: {cache_key}, compressed: {is_compressed}")
            return result is not None

        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            raise CacheError(f"Cache set failed: {e}")

    async def get(self, key: str) -> Optional[T]:
        """
        Get a cache value.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        try:
            cache_key = self._make_key(key)
            raw_data = await self.redis.get(cache_key)

            if raw_data is None:
                return None

            # Check compression marker
            is_compressed = raw_data[0] == 1
            data_without_marker = raw_data[1:]

            # Decompress if needed
            decompressed_data = self._decompress_data(data_without_marker, is_compressed)

            # Deserialize
            value = self.serializer.deserialize(decompressed_data)

            logger.debug(f"Got cache key: {cache_key}, compressed: {is_compressed}")
            return value

        except (SerializationError, zlib.error) as e:
            logger.error(f"Failed to get cache key {key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Cache get failed for key {key}: {e}")
            raise CacheError(f"Cache get failed: {e}")

    async def delete(self, key: str) -> bool:
        """
        Delete a cache key.

        Args:
            key: Cache key

        Returns:
            True if key was deleted
        """
        try:
            cache_key = self._make_key(key)
            result = await self.redis.delete(cache_key)
            logger.debug(f"Deleted cache key: {cache_key}")
            return result > 0
        except Exception as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            raise CacheError(f"Cache delete failed: {e}")

    async def exists(self, key: str) -> bool:
        """Check if a cache key exists."""
        try:
            cache_key = self._make_key(key)
            result = await self.redis.exists(cache_key)
            return result > 0
        except Exception as e:
            logger.error(f"Failed to check cache key existence {key}: {e}")
            raise CacheError(f"Cache exists check failed: {e}")

    async def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for an existing key."""
        try:
            cache_key = self._make_key(key)
            result = await self.redis.expire(cache_key, ttl)
            return result
        except Exception as e:
            logger.error(f"Failed to set expire for cache key {key}: {e}")
            raise CacheError(f"Cache expire failed: {e}")

    async def ttl(self, key: str) -> Optional[int]:
        """Get TTL for a key."""
        try:
            cache_key = self._make_key(key)
            result = await self.redis.ttl(cache_key)
            return result if result >= 0 else None
        except Exception as e:
            logger.error(f"Failed to get TTL for cache key {key}: {e}")
            raise CacheError(f"Cache TTL check failed: {e}")


class RedisManager:
    """
    Redis connection and cache manager for the trading system.

    This class provides:
    - Connection pooling with automatic reconnection
    - Multiple cache managers for different data types
    - Health monitoring and metrics
    - Graceful shutdown
    """

    def __init__(self, config: RedisConfig):
        """
        Initialize Redis manager.

        Args:
            config: Redis configuration
        """
        self.config = config
        self.pool: Optional[ConnectionPool] = None
        self.client: Optional[redis.Redis] = None

        # Cache managers for different data types
        self.funding_rate_cache: Optional[CacheManager[FundingRate]] = None
        self.order_cache: Optional[CacheManager[Order]] = None
        self.opportunity_cache: Optional[CacheManager[ArbitrageOpportunity]] = None
        self.general_cache: Optional[CacheManager[Any]] = None

        # Connection health
        self.is_connected = False
        self._connection_retries = 0
        self._max_retries = 5

    async def initialize(self) -> None:
        """Initialize Redis connection and cache managers."""
        try:
            await self._create_connection()
            await self._setup_cache_managers()
            await self._verify_connection()

            self.is_connected = True
            logger.info("Redis manager initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Redis manager: {e}")
            raise

    async def _create_connection(self) -> None:
        """Create Redis connection pool."""
        try:
            self.pool = ConnectionPool(
                host=self.config.host,
                port=self.config.port,
                db=self.config.db,
                password=self.config.password,
                ssl=self.config.ssl,
                max_connections=20,
                retry_on_timeout=True,
                health_check_interval=30
            )

            self.client = redis.Redis(connection_pool=self.pool)
            logger.info(f"Connected to Redis at {self.config.host}:{self.config.port}")

        except Exception as e:
            logger.error(f"Failed to create Redis connection: {e}")
            raise

    async def _setup_cache_managers(self) -> None:
        """Setup specialized cache managers."""
        if not self.client:
            raise RuntimeError("Redis client not initialized")

        # Funding rates cache - JSON serialization, 5 minute default TTL
        self.funding_rate_cache = CacheManager[FundingRate](
            redis_client=self.client,
            serializer=JSONSerializer(),
            key_prefix="funding_rates",
            default_ttl=300,  # 5 minutes
            compress_threshold=2048
        )

        # Orders cache - Pickle serialization, 1 hour default TTL
        self.order_cache = CacheManager[Order](
            redis_client=self.client,
            serializer=PickleSerializer(),
            key_prefix="orders",
            default_ttl=3600,  # 1 hour
            compress_threshold=1024
        )

        # Arbitrage opportunities cache - JSON serialization, 1 minute default TTL
        self.opportunity_cache = CacheManager[ArbitrageOpportunity](
            redis_client=self.client,
            serializer=JSONSerializer(),
            key_prefix="opportunities",
            default_ttl=60,  # 1 minute
            compress_threshold=1024
        )

        # General purpose cache - JSON serialization, no default TTL
        self.general_cache = CacheManager[Any](
            redis_client=self.client,
            serializer=JSONSerializer(),
            key_prefix="general",
            default_ttl=None,
            compress_threshold=2048
        )

    async def _verify_connection(self) -> None:
        """Verify Redis connection is working."""
        try:
            await self.client.ping()
            logger.info("Redis connection verified")
        except Exception as e:
            raise RuntimeError(f"Redis connection verification failed: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Redis connection.

        Returns:
            Dictionary with health status and metrics
        """
        try:
            start_time = datetime.now()
            await self.client.ping()
            response_time = (datetime.now() - start_time).total_seconds() * 1000

            info = await self.client.info()

            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "redis_version": info.get("redis_version", "unknown"),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0)
            }

        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "response_time_ms": None
            }

    async def get_stats(self) -> Dict[str, Any]:
        """Get Redis statistics and metrics."""
        try:
            info = await self.client.info()
            keyspace_info = await self.client.info("keyspace")

            return {
                "connection": {
                    "connected_clients": info.get("connected_clients", 0),
                    "blocked_clients": info.get("blocked_clients", 0),
                    "tracking_clients": info.get("tracking_clients", 0)
                },
                "memory": {
                    "used_memory": info.get("used_memory", 0),
                    "used_memory_human": info.get("used_memory_human", "0B"),
                    "used_memory_peak": info.get("used_memory_peak", 0),
                    "used_memory_peak_human": info.get("used_memory_peak_human", "0B")
                },
                "stats": {
                    "total_connections_received": info.get("total_connections_received", 0),
                    "total_commands_processed": info.get("total_commands_processed", 0),
                    "keyspace_hits": info.get("keyspace_hits", 0),
                    "keyspace_misses": info.get("keyspace_misses", 0)
                },
                "keyspace": keyspace_info
            }

        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {"error": str(e)}

    async def clear_cache_prefix(self, prefix: str) -> int:
        """
        Clear all keys with a specific prefix.

        Args:
            prefix: Key prefix to clear

        Returns:
            Number of keys deleted
        """
        try:
            pattern = f"{prefix}:*"
            keys = await self.client.keys(pattern)

            if keys:
                deleted = await self.client.delete(*keys)
                logger.info(f"Cleared {deleted} keys with prefix: {prefix}")
                return deleted

            return 0

        except Exception as e:
            logger.error(f"Failed to clear cache prefix {prefix}: {e}")
            raise CacheError(f"Cache clear failed: {e}")

    async def shutdown(self) -> None:
        """Gracefully shutdown Redis connections."""
        try:
            if self.client:
                await self.client.aclose()

            if self.pool:
                await self.pool.aclose()

            self.is_connected = False
            logger.info("Redis manager shutdown completed")

        except Exception as e:
            logger.error(f"Error during Redis shutdown: {e}")
            raise

    def get_funding_rate_cache(self) -> CacheManager[FundingRate]:
        """Get the funding rate cache manager."""
        if not self.funding_rate_cache:
            raise RuntimeError("Redis manager not initialized")
        return self.funding_rate_cache

    def get_order_cache(self) -> CacheManager[Order]:
        """Get the order cache manager."""
        if not self.order_cache:
            raise RuntimeError("Redis manager not initialized")
        return self.order_cache

    def get_opportunity_cache(self) -> CacheManager[ArbitrageOpportunity]:
        """Get the arbitrage opportunity cache manager."""
        if not self.opportunity_cache:
            raise RuntimeError("Redis manager not initialized")
        return self.opportunity_cache

    def get_general_cache(self) -> CacheManager[Any]:
        """Get the general purpose cache manager."""
        if not self.general_cache:
            raise RuntimeError("Redis manager not initialized")
        return self.general_cache