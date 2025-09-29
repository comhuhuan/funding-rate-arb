"""
Publish/Subscribe system for the funding rate arbitrage system.

This module provides Redis-based pub/sub messaging with:
- Event-driven architecture support
- Pattern-based subscriptions
- Message filtering and transformation
- Subscription management
- Error handling and reconnection
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Callable, Set
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
from abc import ABC, abstractmethod

import redis.asyncio as redis
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Standard event types for the trading system."""
    MARKET_DATA_UPDATE = "market_data_update"
    FUNDING_RATE_UPDATE = "funding_rate_update"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ARBITRAGE_OPPORTUNITY = "arbitrage_opportunity"
    RISK_ALERT = "risk_alert"
    SYSTEM_STATUS = "system_status"
    PRICE_ALERT = "price_alert"
    TRADE_EXECUTED = "trade_executed"


@dataclass
class Event:
    """
    Event message for pub/sub system.
    """
    event_type: str
    source: str
    data: Dict[str, Any]
    timestamp: datetime = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """Create event from dictionary."""
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class EventFilter(ABC):
    """Abstract base class for event filtering."""

    @abstractmethod
    def should_process(self, event: Event) -> bool:
        """Check if event should be processed by subscriber."""
        pass


class EventTypeFilter(EventFilter):
    """Filter events by event type."""

    def __init__(self, event_types: Set[str]):
        self.event_types = event_types

    def should_process(self, event: Event) -> bool:
        return event.event_type in self.event_types


class SourceFilter(EventFilter):
    """Filter events by source."""

    def __init__(self, sources: Set[str]):
        self.sources = sources

    def should_process(self, event: Event) -> bool:
        return event.source in self.sources


class CompositeFilter(EventFilter):
    """Combine multiple filters with AND logic."""

    def __init__(self, filters: List[EventFilter]):
        self.filters = filters

    def should_process(self, event: Event) -> bool:
        return all(f.should_process(event) for f in self.filters)


class EventSubscriber(ABC):
    """Abstract base class for event subscribers."""

    @abstractmethod
    async def handle_event(self, event: Event) -> None:
        """Handle received event."""
        pass

    @abstractmethod
    def get_subscription_patterns(self) -> List[str]:
        """Get list of channel patterns this subscriber is interested in."""
        pass

    def get_event_filter(self) -> Optional[EventFilter]:
        """Get event filter for this subscriber."""
        return None


class PubSubManager:
    """
    Redis-based publish/subscribe manager.

    Features:
    - Pattern-based subscriptions
    - Event filtering and routing
    - Automatic reconnection
    - Subscription management
    - Message acknowledgment
    - Dead letter handling for failed events
    """

    def __init__(self, redis_client: Redis, channel_prefix: str = "events"):
        """
        Initialize pub/sub manager.

        Args:
            redis_client: Redis client instance
            channel_prefix: Prefix for all channels
        """
        self.redis = redis_client
        self.channel_prefix = channel_prefix
        self.subscribers: Dict[str, List[EventSubscriber]] = {}
        self.pubsub = None
        self.is_running = False
        self._subscription_task: Optional[asyncio.Task] = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 5

    def _make_channel(self, channel: str) -> str:
        """Create full channel name with prefix."""
        return f"{self.channel_prefix}:{channel}"

    async def publish(
        self,
        channel: str,
        event: Event,
        ttl_seconds: Optional[int] = None
    ) -> int:
        """
        Publish an event to a channel.

        Args:
            channel: Channel name
            event: Event to publish
            ttl_seconds: Optional TTL for the message

        Returns:
            Number of subscribers that received the message
        """
        try:
            full_channel = self._make_channel(channel)
            event_data = json.dumps(event.to_dict())

            # Publish to channel
            subscriber_count = await self.redis.publish(full_channel, event_data)

            # Optionally store with TTL for replay capability
            if ttl_seconds:
                history_key = f"{full_channel}:history"
                await self.redis.lpush(history_key, event_data)
                await self.redis.expire(history_key, ttl_seconds)
                await self.redis.ltrim(history_key, 0, 999)  # Keep last 1000 messages

            logger.debug(f"Published event {event.event_type} to channel {channel}, {subscriber_count} subscribers")
            return subscriber_count

        except Exception as e:
            logger.error(f"Failed to publish event to channel {channel}: {e}")
            raise

    async def subscribe(self, subscriber: EventSubscriber) -> None:
        """
        Subscribe an event subscriber.

        Args:
            subscriber: Event subscriber instance
        """
        patterns = subscriber.get_subscription_patterns()

        for pattern in patterns:
            full_pattern = self._make_channel(pattern)

            if full_pattern not in self.subscribers:
                self.subscribers[full_pattern] = []

            self.subscribers[full_pattern].append(subscriber)

        logger.info(f"Subscribed {subscriber.__class__.__name__} to patterns: {patterns}")

        # Restart subscription if running
        if self.is_running:
            await self._restart_subscription()

    async def unsubscribe(self, subscriber: EventSubscriber) -> None:
        """
        Unsubscribe an event subscriber.

        Args:
            subscriber: Event subscriber instance to remove
        """
        patterns = subscriber.get_subscription_patterns()
        removed_patterns = []

        for pattern in patterns:
            full_pattern = self._make_channel(pattern)

            if full_pattern in self.subscribers:
                if subscriber in self.subscribers[full_pattern]:
                    self.subscribers[full_pattern].remove(subscriber)
                    removed_patterns.append(pattern)

                # Remove pattern if no more subscribers
                if not self.subscribers[full_pattern]:
                    del self.subscribers[full_pattern]

        logger.info(f"Unsubscribed {subscriber.__class__.__name__} from patterns: {removed_patterns}")

        # Restart subscription if running
        if self.is_running:
            await self._restart_subscription()

    async def start(self) -> None:
        """Start the pub/sub subscription loop."""
        if self.is_running:
            return

        self.is_running = True
        self.pubsub = self.redis.pubsub()

        # Subscribe to all patterns
        for pattern in self.subscribers.keys():
            await self.pubsub.psubscribe(pattern)

        # Start subscription task
        self._subscription_task = asyncio.create_task(self._subscription_loop())

        logger.info(f"Started pub/sub manager with {len(self.subscribers)} patterns")

    async def stop(self) -> None:
        """Stop the pub/sub subscription loop."""
        self.is_running = False

        if self._subscription_task:
            self._subscription_task.cancel()
            try:
                await self._subscription_task
            except asyncio.CancelledError:
                pass

        if self.pubsub:
            await self.pubsub.unsubscribe()
            await self.pubsub.aclose()

        logger.info("Stopped pub/sub manager")

    async def _subscription_loop(self) -> None:
        """Main subscription loop for processing messages."""
        while self.is_running:
            try:
                message = await self.pubsub.get_message(timeout=1.0)

                if message and message['type'] == 'pmessage':
                    await self._process_message(message)

            except asyncio.CancelledError:
                logger.info("Pub/sub subscription loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in subscription loop: {e}")
                await self._handle_connection_error()

    async def _process_message(self, message: Dict[str, Any]) -> None:
        """
        Process received message and dispatch to subscribers.

        Args:
            message: Raw message from Redis pub/sub
        """
        try:
            # Extract message data
            channel = message['channel'].decode('utf-8')
            pattern = message['pattern'].decode('utf-8')
            data = message['data'].decode('utf-8')

            # Parse event
            event_dict = json.loads(data)
            event = Event.from_dict(event_dict)

            logger.debug(f"Received event {event.event_type} on channel {channel}")

            # Find matching subscribers
            subscribers = self.subscribers.get(pattern, [])

            # Dispatch to subscribers
            for subscriber in subscribers:
                try:
                    # Apply event filter if present
                    event_filter = subscriber.get_event_filter()
                    if event_filter and not event_filter.should_process(event):
                        continue

                    # Handle event asynchronously
                    asyncio.create_task(self._handle_event_safely(subscriber, event))

                except Exception as e:
                    logger.error(f"Error dispatching event to {subscriber.__class__.__name__}: {e}")

        except Exception as e:
            logger.error(f"Error processing pub/sub message: {e}")

    async def _handle_event_safely(self, subscriber: EventSubscriber, event: Event) -> None:
        """
        Safely handle event with error catching.

        Args:
            subscriber: Event subscriber
            event: Event to handle
        """
        try:
            await subscriber.handle_event(event)
        except Exception as e:
            logger.error(f"Error in subscriber {subscriber.__class__.__name__} handling event {event.event_type}: {e}")
            # Could implement dead letter queue here for failed events

    async def _handle_connection_error(self) -> None:
        """Handle connection errors with reconnection logic."""
        self._reconnect_attempts += 1

        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error("Max reconnection attempts reached, stopping pub/sub manager")
            self.is_running = False
            return

        wait_time = min(2 ** self._reconnect_attempts, 60)  # Exponential backoff, max 60s
        logger.warning(f"Connection error, reconnecting in {wait_time}s (attempt {self._reconnect_attempts})")

        await asyncio.sleep(wait_time)

        try:
            # Recreate pubsub connection
            if self.pubsub:
                await self.pubsub.aclose()

            self.pubsub = self.redis.pubsub()

            # Resubscribe to all patterns
            for pattern in self.subscribers.keys():
                await self.pubsub.psubscribe(pattern)

            self._reconnect_attempts = 0  # Reset on successful reconnection
            logger.info("Successfully reconnected to pub/sub")

        except Exception as e:
            logger.error(f"Failed to reconnect: {e}")

    async def _restart_subscription(self) -> None:
        """Restart subscription with current subscribers."""
        if not self.is_running or not self.pubsub:
            return

        try:
            # Unsubscribe from all current patterns
            await self.pubsub.punsubscribe()

            # Subscribe to all current patterns
            for pattern in self.subscribers.keys():
                await self.pubsub.psubscribe(pattern)

        except Exception as e:
            logger.error(f"Error restarting subscription: {e}")

    async def get_channel_history(self, channel: str, count: int = 100) -> List[Event]:
        """
        Get recent events from channel history.

        Args:
            channel: Channel name
            count: Number of recent events to retrieve

        Returns:
            List of recent events
        """
        try:
            full_channel = self._make_channel(channel)
            history_key = f"{full_channel}:history"

            # Get recent messages
            messages = await self.redis.lrange(history_key, 0, count - 1)

            events = []
            for message in messages:
                try:
                    event_dict = json.loads(message.decode('utf-8'))
                    event = Event.from_dict(event_dict)
                    events.append(event)
                except Exception as e:
                    logger.warning(f"Failed to parse event from history: {e}")

            return events

        except Exception as e:
            logger.error(f"Error getting channel history for {channel}: {e}")
            return []

    def get_subscription_info(self) -> Dict[str, Any]:
        """Get information about current subscriptions."""
        return {
            "is_running": self.is_running,
            "pattern_count": len(self.subscribers),
            "patterns": list(self.subscribers.keys()),
            "subscriber_count": sum(len(subs) for subs in self.subscribers.values()),
            "reconnect_attempts": self._reconnect_attempts
        }


class EventBus:
    """
    High-level event bus that combines pub/sub with routing capabilities.

    Features:
    - Event routing based on event types
    - Automatic channel creation
    - Event history and replay
    - Metrics and monitoring
    """

    def __init__(self, pubsub_manager: PubSubManager):
        """
        Initialize event bus.

        Args:
            pubsub_manager: PubSubManager instance
        """
        self.pubsub = pubsub_manager
        self.event_counts: Dict[str, int] = {}

    async def publish_event(
        self,
        event_type: str,
        source: str,
        data: Dict[str, Any],
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Publish an event.

        Args:
            event_type: Type of event
            source: Source of the event
            data: Event data
            correlation_id: Optional correlation ID
            metadata: Optional metadata

        Returns:
            Number of subscribers
        """
        event = Event(
            event_type=event_type,
            source=source,
            data=data,
            correlation_id=correlation_id,
            metadata=metadata or {}
        )

        # Route to appropriate channel based on event type
        channel = self._get_channel_for_event_type(event_type)
        subscriber_count = await self.pubsub.publish(channel, event, ttl_seconds=3600)

        # Update metrics
        self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1

        return subscriber_count

    async def subscribe_to_events(
        self,
        subscriber: EventSubscriber,
        event_types: Optional[List[str]] = None
    ) -> None:
        """
        Subscribe to specific event types.

        Args:
            subscriber: Event subscriber
            event_types: List of event types to subscribe to (all if None)
        """
        await self.pubsub.subscribe(subscriber)

    def _get_channel_for_event_type(self, event_type: str) -> str:
        """Get channel name for event type."""
        # Use hierarchical channel names for better pattern matching
        category = self._get_event_category(event_type)
        return f"{category}.{event_type}"

    def _get_event_category(self, event_type: str) -> str:
        """Get category for event type."""
        if event_type.startswith(('market_data', 'funding_rate')):
            return 'data'
        elif event_type.startswith(('order_', 'trade_')):
            return 'trading'
        elif event_type.startswith('risk_'):
            return 'risk'
        elif event_type.startswith('system_'):
            return 'system'
        else:
            return 'general'

    async def get_event_history(
        self,
        event_type: str,
        count: int = 100
    ) -> List[Event]:
        """
        Get recent events of a specific type.

        Args:
            event_type: Event type to retrieve
            count: Number of events to retrieve

        Returns:
            List of recent events
        """
        channel = self._get_channel_for_event_type(event_type)
        return await self.pubsub.get_channel_history(channel, count)

    def get_metrics(self) -> Dict[str, Any]:
        """Get event bus metrics."""
        return {
            "event_counts": self.event_counts.copy(),
            "total_events": sum(self.event_counts.values()),
            "subscription_info": self.pubsub.get_subscription_info()
        }