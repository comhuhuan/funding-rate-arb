"""
Unit tests for the infrastructure layer components.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Optional

from src.infrastructure.cache.redis_manager import (
    RedisManager, CacheManager, JSONSerializer, PickleSerializer,
    SerializationError, CacheError
)
from src.infrastructure.messaging.message_queue import (
    MessageQueue, MessageBroker, Message, MessagePriority, MessageStatus,
    MessageHandler
)
from src.infrastructure.messaging.pubsub import (
    PubSubManager, EventBus, Event, EventType, EventFilter,
    EventTypeFilter, SourceFilter, CompositeFilter, EventSubscriber
)
from src.infrastructure.config.manager import RedisConfig
from src.types import FundingRate, Order, Symbol, OrderSide, OrderType, OrderStatus


class TestSerializers:
    """Test serialization classes."""

    def test_json_serializer_basic_types(self):
        """Test JSONSerializer with basic types."""
        serializer = JSONSerializer()

        # Test basic data
        data = {"key": "value", "number": 123, "boolean": True}
        serialized = serializer.serialize(data)
        deserialized = serializer.deserialize(serialized)

        assert deserialized == data

    def test_json_serializer_decimal_support(self):
        """Test JSONSerializer with Decimal types."""
        serializer = JSONSerializer()

        # Test that decimals are handled correctly
        data = {"price": Decimal("123.45"), "amount": Decimal("100.0")}
        serialized = serializer.serialize(data)
        deserialized = serializer.deserialize(serialized)

        # Verify the serialization created the special decimal format
        json_str = serialized.decode('utf-8')
        assert '__decimal__' in json_str

        # After deserialization, decimals should be restored
        assert isinstance(deserialized["price"], Decimal)
        assert deserialized["price"] == Decimal("123.45")
        assert isinstance(deserialized["amount"], Decimal)
        assert deserialized["amount"] == Decimal("100.0")

    def test_json_serializer_error_handling(self):
        """Test JSONSerializer error handling."""
        serializer = JSONSerializer()

        # Test deserialization error with invalid JSON
        with pytest.raises(SerializationError):
            serializer.deserialize(b"invalid json")

        # Test deserialization error with invalid UTF-8
        with pytest.raises(SerializationError):
            serializer.deserialize(b'\xff\xfe')

    def test_pickle_serializer(self):
        """Test PickleSerializer."""
        serializer = PickleSerializer()

        # Test with complex object
        data = {"nested": {"list": [1, 2, 3], "set": {4, 5, 6}}}
        serialized = serializer.serialize(data)
        deserialized = serializer.deserialize(serialized)

        assert deserialized == data


class TestCacheManager:
    """Test CacheManager functionality."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        redis_mock = AsyncMock()
        redis_mock.get.return_value = None
        redis_mock.set.return_value = True
        redis_mock.delete.return_value = 1
        redis_mock.exists.return_value = 1
        return redis_mock

    @pytest.fixture
    def cache_manager(self, mock_redis):
        """Create CacheManager instance."""
        return CacheManager(
            redis_client=mock_redis,
            serializer=JSONSerializer(),
            key_prefix="test",
            default_ttl=300
        )

    @pytest.mark.asyncio
    async def test_cache_set_get(self, cache_manager, mock_redis):
        """Test basic set/get operations."""
        test_data = {"key": "value", "number": 123}

        # Test set
        result = await cache_manager.set("test_key", test_data)
        assert result is True

        # Verify Redis was called correctly
        mock_redis.set.assert_called_once()
        args, kwargs = mock_redis.set.call_args
        assert args[0] == "test:test_key"
        assert kwargs["ex"] == 300

        # Test get
        serialized_data = json.dumps(test_data).encode('utf-8')
        compressed_data = b'\x00' + serialized_data  # No compression marker
        mock_redis.get.return_value = compressed_data

        result = await cache_manager.get("test_key")
        assert result == test_data

    @pytest.mark.asyncio
    async def test_cache_compression(self, cache_manager, mock_redis):
        """Test data compression for large objects."""
        # Create large data that exceeds compression threshold
        large_data = {"data": "x" * 2000}  # Larger than 1024 byte threshold

        await cache_manager.set("large_key", large_data)

        # Verify compression marker was added
        args, kwargs = mock_redis.set.call_args
        data = args[1]
        assert data[0] == 1  # Compression marker

    @pytest.mark.asyncio
    async def test_cache_ttl_operations(self, cache_manager, mock_redis):
        """Test TTL-related operations."""
        # Test expire
        mock_redis.expire.return_value = True
        result = await cache_manager.expire("test_key", 600)
        assert result is True
        mock_redis.expire.assert_called_with("test:test_key", 600)

        # Test ttl
        mock_redis.ttl.return_value = 300
        result = await cache_manager.ttl("test_key")
        assert result == 300

    @pytest.mark.asyncio
    async def test_cache_error_handling(self, cache_manager, mock_redis):
        """Test error handling."""
        # Test Redis error
        mock_redis.set.side_effect = Exception("Redis error")

        with pytest.raises(CacheError):
            await cache_manager.set("test_key", {"data": "test"})

    @pytest.mark.asyncio
    async def test_cache_serialization_error(self, cache_manager, mock_redis):
        """Test serialization error handling."""
        # Mock get to return invalid data
        mock_redis.get.return_value = b'\x00invalid_json'

        result = await cache_manager.get("test_key")
        assert result is None  # Should return None on deserialization error


class TestRedisManager:
    """Test RedisManager functionality."""

    @pytest.fixture
    def redis_config(self):
        """Create RedisConfig for testing."""
        return RedisConfig(
            host="localhost",
            port=6379,
            db=0,
            password=None,
            ssl=False
        )

    @pytest.fixture
    def mock_redis_client(self):
        """Mock Redis client."""
        mock_client = AsyncMock()
        mock_client.ping.return_value = b'PONG'
        mock_client.info.return_value = {
            'connected_clients': 1,
            'used_memory': 1024,
            'used_memory_human': '1K',
            'redis_version': '6.0.0',
            'uptime_in_seconds': 3600
        }
        return mock_client

    @patch('src.infrastructure.cache.redis_manager.ConnectionPool')
    @patch('src.infrastructure.cache.redis_manager.redis.Redis')
    @pytest.mark.asyncio
    async def test_redis_manager_initialization(self, mock_redis_class, mock_pool_class, redis_config):
        """Test RedisManager initialization."""
        mock_redis_instance = AsyncMock()
        mock_redis_instance.ping.return_value = b'PONG'
        mock_redis_class.return_value = mock_redis_instance

        manager = RedisManager(redis_config)
        await manager.initialize()

        assert manager.is_connected is True
        assert manager.funding_rate_cache is not None
        assert manager.order_cache is not None
        assert manager.opportunity_cache is not None
        assert manager.general_cache is not None

    @pytest.mark.asyncio
    async def test_redis_manager_health_check(self, redis_config, mock_redis_client):
        """Test Redis health check."""
        manager = RedisManager(redis_config)
        manager.client = mock_redis_client

        health = await manager.health_check()

        assert health["status"] == "healthy"
        assert "response_time_ms" in health
        assert health["connected_clients"] == 1
        assert health["redis_version"] == "6.0.0"

    @pytest.mark.asyncio
    async def test_redis_manager_health_check_failure(self, redis_config):
        """Test Redis health check failure."""
        manager = RedisManager(redis_config)
        mock_client = AsyncMock()
        mock_client.ping.side_effect = Exception("Connection failed")
        manager.client = mock_client

        health = await manager.health_check()

        assert health["status"] == "unhealthy"
        assert "error" in health


class TestMessage:
    """Test Message class."""

    def test_message_creation(self):
        """Test message creation with defaults."""
        payload = {"type": "test", "data": "value"}
        message = Message(
            id="test-123",
            queue_name="test_queue",
            payload=payload
        )

        assert message.id == "test-123"
        assert message.queue_name == "test_queue"
        assert message.payload == payload
        assert message.priority == MessagePriority.NORMAL
        assert message.status == MessageStatus.PENDING
        assert message.retry_count == 0
        assert message.created_at is not None

    def test_message_serialization(self):
        """Test message to_dict and from_dict."""
        payload = {"type": "test"}
        message = Message(
            id="test-123",
            queue_name="test_queue",
            payload=payload,
            priority=MessagePriority.HIGH
        )

        # Test to_dict
        data = message.to_dict()
        assert data["id"] == "test-123"
        assert data["priority"] == MessagePriority.HIGH.value
        assert data["status"] == MessageStatus.PENDING.value
        assert "created_at" in data

        # Test from_dict
        restored_message = Message.from_dict(data)
        assert restored_message.id == message.id
        assert restored_message.priority == message.priority
        assert restored_message.status == message.status

    def test_message_retry_logic(self):
        """Test message retry logic."""
        message = Message(
            id="test-123",
            queue_name="test_queue",
            payload={},
            max_retries=3
        )

        # Should not retry when status is not FAILED
        assert message.should_retry() is False

        # Should retry when failed and under max retries
        message.status = MessageStatus.FAILED
        message.retry_count = 1
        assert message.should_retry() is True

        # Should not retry when max retries reached
        message.retry_count = 3
        assert message.should_retry() is False

    def test_message_expiry(self):
        """Test message expiry logic."""
        # Create message that expires in 1 second
        message = Message(
            id="test-123",
            queue_name="test_queue",
            payload={},
            timeout_seconds=1,
            created_at=datetime.utcnow() - timedelta(seconds=2)
        )

        assert message.is_expired() is True

        # Test message that doesn't expire
        message.timeout_seconds = 0
        assert message.is_expired() is False


class TestMessageQueue:
    """Test MessageQueue functionality."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for MessageQueue tests."""
        redis_mock = AsyncMock()
        redis_mock.lpush.return_value = 1
        redis_mock.zadd.return_value = 1
        redis_mock.brpoplpush.return_value = None
        redis_mock.lrem.return_value = 1
        redis_mock.llen.return_value = 0
        redis_mock.zcard.return_value = 0
        redis_mock.hincrby.return_value = 1
        return redis_mock

    @pytest.fixture
    def message_queue(self, mock_redis):
        """Create MessageQueue instance."""
        return MessageQueue(
            redis_client=mock_redis,
            queue_name="test_queue",
            max_retries=3,
            retry_delay_seconds=60
        )

    @pytest.mark.asyncio
    async def test_enqueue_normal_message(self, message_queue, mock_redis):
        """Test enqueuing normal priority message."""
        payload = {"type": "test", "data": "value"}

        message_id = await message_queue.enqueue(payload)

        assert message_id is not None
        mock_redis.lpush.assert_called_once()

        # Verify correct queue was used
        args = mock_redis.lpush.call_args[0]
        assert "queue:test_queue:p2" in args[0]  # Priority 2 = NORMAL

    @pytest.mark.asyncio
    async def test_enqueue_delayed_message(self, message_queue, mock_redis):
        """Test enqueuing delayed message."""
        payload = {"type": "test"}

        message_id = await message_queue.enqueue(payload, delay_seconds=300)

        assert message_id is not None
        mock_redis.zadd.assert_called_once()

        # Verify delayed queue was used
        args = mock_redis.zadd.call_args[0]
        assert "queue:test_queue:delayed" in args[0]

    @pytest.mark.asyncio
    async def test_dequeue_message(self, message_queue, mock_redis):
        """Test dequeuing message."""
        # Mock message data
        test_message = Message(
            id="test-123",
            queue_name="test_queue",
            payload={"type": "test"}
        )
        message_json = json.dumps(test_message.to_dict())
        mock_redis.brpoplpush.return_value = message_json.encode('utf-8')

        # Mock delayed message processing
        mock_redis.zrangebyscore.return_value = []

        message = await message_queue.dequeue(timeout=1)

        assert message is not None
        assert message.id == "test-123"
        assert message.status == MessageStatus.PROCESSING

    @pytest.mark.asyncio
    async def test_ack_successful_message(self, message_queue, mock_redis):
        """Test acknowledging successful message."""
        message = Message(
            id="test-123",
            queue_name="test_queue",
            payload={"type": "test"}
        )

        await message_queue.ack(message, success=True)

        mock_redis.lrem.assert_called_once()
        assert message.status == MessageStatus.COMPLETED
        assert message.processed_at is not None

    @pytest.mark.asyncio
    async def test_ack_failed_message_with_retry(self, message_queue, mock_redis):
        """Test acknowledging failed message that should retry."""
        message = Message(
            id="test-123",
            queue_name="test_queue",
            payload={"type": "test"},
            max_retries=3,
            retry_count=1
        )

        await message_queue.ack(message, success=False, error_message="Test error")

        assert message.status == MessageStatus.PENDING  # Reset for retry
        assert message.retry_count == 2
        assert message.error_message == "Test error"
        mock_redis.zadd.assert_called_once()  # Added to delayed queue for retry

    @pytest.mark.asyncio
    async def test_ack_failed_message_dead_letter(self, message_queue, mock_redis):
        """Test failed message moved to dead letter queue."""
        message = Message(
            id="test-123",
            queue_name="test_queue",
            payload={"type": "test"},
            max_retries=3,
            retry_count=3  # At max retries
        )

        await message_queue.ack(message, success=False)

        assert message.status == MessageStatus.DEAD_LETTER
        mock_redis.lpush.assert_called()  # Added to dead letter queue

    @pytest.mark.asyncio
    async def test_queue_size(self, message_queue, mock_redis):
        """Test getting queue sizes."""
        mock_redis.llen.return_value = 5
        mock_redis.zcard.return_value = 2

        sizes = await message_queue.size()

        assert "total" in sizes
        assert "delayed" in sizes
        assert "processing" in sizes
        assert "dead_letter" in sizes


class MockMessageHandler(MessageHandler):
    """Mock message handler for testing."""

    def __init__(self, queue_name: str, should_succeed: bool = True):
        self.queue_name = queue_name
        self.should_succeed = should_succeed
        self.handled_messages = []

    async def handle(self, message: Message) -> bool:
        self.handled_messages.append(message)
        return self.should_succeed

    def can_handle(self, queue_name: str) -> bool:
        return queue_name == self.queue_name


class TestMessageBroker:
    """Test MessageBroker functionality."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        return AsyncMock()

    @pytest.fixture
    def message_broker(self, mock_redis):
        """Create MessageBroker instance."""
        return MessageBroker(mock_redis)

    def test_create_queue(self, message_broker):
        """Test queue creation."""
        queue = message_broker.create_queue("test_queue")

        assert queue is not None
        assert "test_queue" in message_broker.queues
        assert "test_queue" in message_broker.handlers

        # Test getting existing queue
        same_queue = message_broker.create_queue("test_queue")
        assert same_queue is queue

    def test_register_handler(self, message_broker):
        """Test handler registration."""
        message_broker.create_queue("test_queue")
        handler = MockMessageHandler("test_queue")

        message_broker.register_handler("test_queue", handler)

        assert handler in message_broker.handlers["test_queue"]

    @pytest.mark.asyncio
    async def test_publish_message(self, message_broker, mock_redis):
        """Test publishing message."""
        queue = message_broker.create_queue("test_queue")

        # Mock the enqueue method
        queue.enqueue = AsyncMock(return_value="test-message-id")

        payload = {"type": "test", "data": "value"}
        message_id = await message_broker.publish("test_queue", payload)

        assert message_id == "test-message-id"
        queue.enqueue.assert_called_once_with(payload, MessagePriority.NORMAL)

    def test_publish_nonexistent_queue(self, message_broker):
        """Test publishing to non-existent queue."""
        with pytest.raises(ValueError, match="Queue .* does not exist"):
            asyncio.run(message_broker.publish("nonexistent", {}))


class TestEvent:
    """Test Event class."""

    def test_event_creation(self):
        """Test event creation with defaults."""
        data = {"key": "value"}
        event = Event(
            event_type="test_event",
            source="test_source",
            data=data
        )

        assert event.event_type == "test_event"
        assert event.source == "test_source"
        assert event.data == data
        assert event.timestamp is not None
        assert event.metadata == {}

    def test_event_serialization(self):
        """Test event serialization."""
        data = {"key": "value"}
        event = Event(
            event_type="test_event",
            source="test_source",
            data=data,
            correlation_id="test-123"
        )

        # Test to_dict
        event_dict = event.to_dict()
        assert event_dict["event_type"] == "test_event"
        assert event_dict["correlation_id"] == "test-123"
        assert isinstance(event_dict["timestamp"], str)

        # Test from_dict
        restored_event = Event.from_dict(event_dict)
        assert restored_event.event_type == event.event_type
        assert restored_event.source == event.source
        assert restored_event.data == event.data
        assert isinstance(restored_event.timestamp, datetime)


class TestEventFilters:
    """Test event filtering classes."""

    def test_event_type_filter(self):
        """Test EventTypeFilter."""
        filter_obj = EventTypeFilter({"type1", "type2"})

        event1 = Event("type1", "source", {})
        event2 = Event("type3", "source", {})

        assert filter_obj.should_process(event1) is True
        assert filter_obj.should_process(event2) is False

    def test_source_filter(self):
        """Test SourceFilter."""
        filter_obj = SourceFilter({"source1", "source2"})

        event1 = Event("type", "source1", {})
        event2 = Event("type", "source3", {})

        assert filter_obj.should_process(event1) is True
        assert filter_obj.should_process(event2) is False

    def test_composite_filter(self):
        """Test CompositeFilter."""
        type_filter = EventTypeFilter({"type1"})
        source_filter = SourceFilter({"source1"})
        composite = CompositeFilter([type_filter, source_filter])

        event1 = Event("type1", "source1", {})  # Matches both filters
        event2 = Event("type1", "source2", {})  # Matches only type filter
        event3 = Event("type2", "source1", {})  # Matches only source filter

        assert composite.should_process(event1) is True
        assert composite.should_process(event2) is False
        assert composite.should_process(event3) is False


class MockEventSubscriber(EventSubscriber):
    """Mock event subscriber for testing."""

    def __init__(self, patterns: List[str], event_filter: EventFilter = None):
        self.patterns = patterns
        self.event_filter = event_filter
        self.received_events = []

    async def handle_event(self, event: Event) -> None:
        self.received_events.append(event)

    def get_subscription_patterns(self) -> List[str]:
        return self.patterns

    def get_event_filter(self) -> Optional[EventFilter]:
        return self.event_filter


class TestPubSubManager:
    """Test PubSubManager functionality."""

    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        redis_mock = AsyncMock()
        redis_mock.publish.return_value = 1
        redis_mock.pubsub.return_value = AsyncMock()
        return redis_mock

    @pytest.fixture
    def pubsub_manager(self, mock_redis):
        """Create PubSubManager instance."""
        return PubSubManager(mock_redis, channel_prefix="test")

    @pytest.mark.asyncio
    async def test_publish_event(self, pubsub_manager, mock_redis):
        """Test publishing event."""
        event = Event(
            event_type="test_event",
            source="test_source",
            data={"key": "value"}
        )

        subscriber_count = await pubsub_manager.publish("test_channel", event)

        assert subscriber_count == 1
        mock_redis.publish.assert_called_once()

        # Verify channel name
        args = mock_redis.publish.call_args[0]
        assert args[0] == "test:test_channel"

    @pytest.mark.asyncio
    async def test_subscribe_subscriber(self, pubsub_manager):
        """Test subscribing event subscriber."""
        subscriber = MockEventSubscriber(["pattern1", "pattern2"])

        await pubsub_manager.subscribe(subscriber)

        assert "test:pattern1" in pubsub_manager.subscribers
        assert "test:pattern2" in pubsub_manager.subscribers
        assert subscriber in pubsub_manager.subscribers["test:pattern1"]

    @pytest.mark.asyncio
    async def test_unsubscribe_subscriber(self, pubsub_manager):
        """Test unsubscribing event subscriber."""
        subscriber = MockEventSubscriber(["pattern1"])

        await pubsub_manager.subscribe(subscriber)
        await pubsub_manager.unsubscribe(subscriber)

        assert "test:pattern1" not in pubsub_manager.subscribers

    def test_subscription_info(self, pubsub_manager):
        """Test getting subscription information."""
        info = pubsub_manager.get_subscription_info()

        assert "is_running" in info
        assert "pattern_count" in info
        assert "patterns" in info
        assert "subscriber_count" in info


class TestEventBus:
    """Test EventBus functionality."""

    @pytest.fixture
    def mock_pubsub(self):
        """Mock PubSubManager."""
        pubsub_mock = AsyncMock()
        pubsub_mock.publish.return_value = 2
        return pubsub_mock

    @pytest.fixture
    def event_bus(self, mock_pubsub):
        """Create EventBus instance."""
        return EventBus(mock_pubsub)

    @pytest.mark.asyncio
    async def test_publish_event(self, event_bus, mock_pubsub):
        """Test publishing event through event bus."""
        subscriber_count = await event_bus.publish_event(
            event_type="market_data_update",
            source="exchange_adapter",
            data={"symbol": "BTC/USD", "price": 50000}
        )

        assert subscriber_count == 2
        mock_pubsub.publish.assert_called_once()

        # Verify channel routing
        args = mock_pubsub.publish.call_args[0]
        assert args[0] == "data.market_data_update"  # Category-based routing

    def test_event_categorization(self, event_bus):
        """Test event category routing."""
        # Test data category
        channel = event_bus._get_channel_for_event_type("market_data_update")
        assert channel == "data.market_data_update"

        # Test trading category
        channel = event_bus._get_channel_for_event_type("order_placed")
        assert channel == "trading.order_placed"

        # Test risk category
        channel = event_bus._get_channel_for_event_type("risk_alert")
        assert channel == "risk.risk_alert"

        # Test system category
        channel = event_bus._get_channel_for_event_type("system_status")
        assert channel == "system.system_status"

        # Test general category
        channel = event_bus._get_channel_for_event_type("custom_event")
        assert channel == "general.custom_event"

    def test_get_metrics(self, event_bus):
        """Test getting event bus metrics."""
        metrics = event_bus.get_metrics()

        assert "event_counts" in metrics
        assert "total_events" in metrics
        assert "subscription_info" in metrics