"""
Message queue system for the funding rate arbitrage system.

This module provides Redis-based message queuing capabilities with:
- FIFO and priority queue support
- Dead letter queues for failed messages
- Message persistence and acknowledgment
- Retry mechanisms with exponential backoff
- Metrics and monitoring
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

import redis.asyncio as redis
from redis.asyncio import Redis

logger = logging.getLogger(__name__)

T = TypeVar('T')


class MessagePriority(Enum):
    """Message priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class MessageStatus(Enum):
    """Message processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


@dataclass
class Message:
    """
    Message wrapper for queue operations.
    """
    id: str
    queue_name: str
    payload: Dict[str, Any]
    priority: MessagePriority = MessagePriority.NORMAL
    created_at: datetime = None
    scheduled_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    status: MessageStatus = MessageStatus.PENDING
    error_message: Optional[str] = None
    processed_at: Optional[datetime] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.id is None:
            self.id = str(uuid.uuid4())

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO format strings
        for field in ['created_at', 'scheduled_at', 'processed_at']:
            if data[field] is not None:
                data[field] = data[field].isoformat()
        # Convert enums to values
        data['priority'] = self.priority.value
        data['status'] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary."""
        # Convert datetime strings back to datetime objects
        for field in ['created_at', 'scheduled_at', 'processed_at']:
            if data.get(field):
                data[field] = datetime.fromisoformat(data[field])

        # Convert enum values back to enums
        if 'priority' in data:
            data['priority'] = MessagePriority(data['priority'])
        if 'status' in data:
            data['status'] = MessageStatus(data['status'])

        return cls(**data)

    def should_retry(self) -> bool:
        """Check if message should be retried."""
        return self.retry_count < self.max_retries and self.status == MessageStatus.FAILED

    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.timeout_seconds <= 0:
            return False
        expiry_time = self.created_at + timedelta(seconds=self.timeout_seconds)
        return datetime.utcnow() > expiry_time


class MessageHandler(ABC):
    """Abstract base class for message handlers."""

    @abstractmethod
    async def handle(self, message: Message) -> bool:
        """
        Handle a message.

        Args:
            message: The message to process

        Returns:
            True if message was processed successfully, False otherwise
        """
        pass

    @abstractmethod
    def can_handle(self, queue_name: str) -> bool:
        """Check if this handler can process messages from the given queue."""
        pass


class MessageQueue:
    """
    Redis-based message queue with advanced features.

    Features:
    - Priority queues with multiple priority levels
    - Delayed message delivery
    - Message acknowledgment and retry logic
    - Dead letter queue for failed messages
    - Metrics and monitoring
    """

    def __init__(
        self,
        redis_client: Redis,
        queue_name: str,
        dead_letter_queue: Optional[str] = None,
        max_retries: int = 3,
        retry_delay_seconds: int = 60
    ):
        """
        Initialize message queue.

        Args:
            redis_client: Redis client instance
            queue_name: Name of the queue
            dead_letter_queue: Name of dead letter queue
            max_retries: Maximum retry attempts
            retry_delay_seconds: Base delay between retries
        """
        self.redis = redis_client
        self.queue_name = queue_name
        self.dead_letter_queue = dead_letter_queue or f"{queue_name}:dlq"
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

        # Redis key names
        self.queue_key = f"queue:{queue_name}"
        self.processing_key = f"queue:{queue_name}:processing"
        self.delayed_key = f"queue:{queue_name}:delayed"
        self.dlq_key = f"queue:{dead_letter_queue}"
        self.stats_key = f"queue:{queue_name}:stats"

    async def enqueue(
        self,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        delay_seconds: int = 0,
        **kwargs
    ) -> str:
        """
        Add a message to the queue.

        Args:
            payload: Message payload
            priority: Message priority
            delay_seconds: Delay before message is available
            **kwargs: Additional message properties

        Returns:
            Message ID
        """
        message = Message(
            id=str(uuid.uuid4()),
            queue_name=self.queue_name,
            payload=payload,
            priority=priority,
            scheduled_at=datetime.utcnow() + timedelta(seconds=delay_seconds) if delay_seconds > 0 else None,
            max_retries=kwargs.get('max_retries', self.max_retries),
            timeout_seconds=kwargs.get('timeout_seconds', 300),
            **{k: v for k, v in kwargs.items() if k not in ['max_retries', 'timeout_seconds']}
        )

        message_data = json.dumps(message.to_dict())

        if delay_seconds > 0:
            # Add to delayed queue with score as timestamp
            score = (datetime.utcnow() + timedelta(seconds=delay_seconds)).timestamp()
            await self.redis.zadd(self.delayed_key, {message_data: score})
        else:
            # Add to priority queue
            queue_key = f"{self.queue_key}:p{priority.value}"
            await self.redis.lpush(queue_key, message_data)

        # Update stats
        await self._increment_stat("enqueued")

        logger.debug(f"Enqueued message {message.id} to queue {self.queue_name}")
        return message.id

    async def dequeue(self, timeout: int = 10) -> Optional[Message]:
        """
        Remove and return a message from the queue.

        Args:
            timeout: Timeout in seconds for blocking operation

        Returns:
            Message or None if timeout
        """
        # First, process any delayed messages that are ready
        await self._process_delayed_messages()

        # Try to get message from priority queues (highest priority first)
        for priority in reversed(MessagePriority):
            queue_key = f"{self.queue_key}:p{priority.value}"
            result = await self.redis.brpoplpush(queue_key, self.processing_key, timeout=1)

            if result:
                message_data = result.decode('utf-8')
                message = Message.from_dict(json.loads(message_data))
                message.status = MessageStatus.PROCESSING

                logger.debug(f"Dequeued message {message.id} from queue {self.queue_name}")
                await self._increment_stat("dequeued")
                return message

        return None

    async def ack(self, message: Message, success: bool = True, error_message: Optional[str] = None) -> None:
        """
        Acknowledge message processing.

        Args:
            message: The processed message
            success: Whether processing was successful
            error_message: Error message if processing failed
        """
        message_data = json.dumps(message.to_dict())

        # Remove from processing queue
        await self.redis.lrem(self.processing_key, 1, message_data)

        if success:
            message.status = MessageStatus.COMPLETED
            message.processed_at = datetime.utcnow()
            await self._increment_stat("completed")
            logger.debug(f"Acknowledged successful processing of message {message.id}")

        else:
            message.status = MessageStatus.FAILED
            message.retry_count += 1
            message.error_message = error_message

            if message.should_retry():
                # Schedule retry with exponential backoff
                delay = self.retry_delay_seconds * (2 ** (message.retry_count - 1))
                retry_time = datetime.utcnow() + timedelta(seconds=delay)

                message.scheduled_at = retry_time
                message.status = MessageStatus.PENDING

                retry_data = json.dumps(message.to_dict())
                await self.redis.zadd(self.delayed_key, {retry_data: retry_time.timestamp()})

                await self._increment_stat("retried")
                logger.info(f"Scheduled retry for message {message.id} in {delay} seconds")

            else:
                # Move to dead letter queue
                message.status = MessageStatus.DEAD_LETTER
                dlq_data = json.dumps(message.to_dict())
                await self.redis.lpush(self.dlq_key, dlq_data)

                await self._increment_stat("dead_lettered")
                logger.warning(f"Moved message {message.id} to dead letter queue after {message.retry_count} retries")

    async def peek(self, count: int = 1) -> List[Message]:
        """
        Peek at messages without removing them from queue.

        Args:
            count: Number of messages to peek at

        Returns:
            List of messages
        """
        messages = []

        for priority in reversed(MessagePriority):
            queue_key = f"{self.queue_key}:p{priority.value}"
            items = await self.redis.lrange(queue_key, -count, -1)

            for item in items:
                message_data = json.loads(item.decode('utf-8'))
                messages.append(Message.from_dict(message_data))

            if len(messages) >= count:
                break

        return messages[:count]

    async def size(self) -> Dict[str, int]:
        """
        Get queue sizes.

        Returns:
            Dictionary with queue sizes for each priority
        """
        sizes = {}
        total = 0

        for priority in MessagePriority:
            queue_key = f"{self.queue_key}:p{priority.value}"
            size = await self.redis.llen(queue_key)
            sizes[priority.name.lower()] = size
            total += size

        # Add delayed and processing queue sizes
        sizes['delayed'] = await self.redis.zcard(self.delayed_key)
        sizes['processing'] = await self.redis.llen(self.processing_key)
        sizes['dead_letter'] = await self.redis.llen(self.dlq_key)
        sizes['total'] = total

        return sizes

    async def purge(self, include_dlq: bool = False) -> int:
        """
        Clear all messages from the queue.

        Args:
            include_dlq: Whether to also clear dead letter queue

        Returns:
            Number of messages removed
        """
        count = 0

        # Clear priority queues
        for priority in MessagePriority:
            queue_key = f"{self.queue_key}:p{priority.value}"
            queue_size = await self.redis.llen(queue_key)
            if queue_size > 0:
                await self.redis.delete(queue_key)
                count += queue_size

        # Clear delayed and processing queues
        count += await self.redis.zcard(self.delayed_key)
        count += await self.redis.llen(self.processing_key)
        await self.redis.delete(self.delayed_key, self.processing_key)

        if include_dlq:
            count += await self.redis.llen(self.dlq_key)
            await self.redis.delete(self.dlq_key)

        logger.info(f"Purged {count} messages from queue {self.queue_name}")
        return count

    async def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        stats = await self.redis.hgetall(self.stats_key)
        sizes = await self.size()

        return {
            "queue_name": self.queue_name,
            "sizes": sizes,
            "stats": {k.decode('utf-8'): int(v) for k, v in stats.items()},
            "created_at": datetime.utcnow().isoformat()
        }

    async def _process_delayed_messages(self) -> None:
        """Process delayed messages that are ready."""
        now = datetime.utcnow().timestamp()

        # Get ready messages
        ready_messages = await self.redis.zrangebyscore(
            self.delayed_key, 0, now, withscores=False
        )

        for message_data in ready_messages:
            # Remove from delayed queue
            await self.redis.zrem(self.delayed_key, message_data)

            # Parse message and add to appropriate priority queue
            message_dict = json.loads(message_data.decode('utf-8'))
            message = Message.from_dict(message_dict)

            queue_key = f"{self.queue_key}:p{message.priority.value}"
            await self.redis.lpush(queue_key, message_data)

    async def _increment_stat(self, stat_name: str) -> None:
        """Increment a queue statistic."""
        await self.redis.hincrby(self.stats_key, stat_name, 1)


class MessageBroker:
    """
    Message broker that manages multiple queues and message processing.

    Features:
    - Multiple queue management
    - Message routing
    - Handler registration and dispatch
    - Worker pool management
    - Metrics aggregation
    """

    def __init__(self, redis_client: Redis):
        """
        Initialize message broker.

        Args:
            redis_client: Redis client instance
        """
        self.redis = redis_client
        self.queues: Dict[str, MessageQueue] = {}
        self.handlers: Dict[str, List[MessageHandler]] = {}
        self.workers: Dict[str, asyncio.Task] = {}
        self.is_running = False

    def create_queue(
        self,
        queue_name: str,
        dead_letter_queue: Optional[str] = None,
        max_retries: int = 3,
        retry_delay_seconds: int = 60
    ) -> MessageQueue:
        """
        Create a new message queue.

        Args:
            queue_name: Name of the queue
            dead_letter_queue: Name of dead letter queue
            max_retries: Maximum retry attempts
            retry_delay_seconds: Base delay between retries

        Returns:
            Created MessageQueue instance
        """
        if queue_name in self.queues:
            return self.queues[queue_name]

        queue = MessageQueue(
            redis_client=self.redis,
            queue_name=queue_name,
            dead_letter_queue=dead_letter_queue,
            max_retries=max_retries,
            retry_delay_seconds=retry_delay_seconds
        )

        self.queues[queue_name] = queue
        self.handlers[queue_name] = []

        logger.info(f"Created message queue: {queue_name}")
        return queue

    def register_handler(self, queue_name: str, handler: MessageHandler) -> None:
        """
        Register a message handler for a queue.

        Args:
            queue_name: Name of the queue
            handler: Message handler instance
        """
        if queue_name not in self.handlers:
            self.handlers[queue_name] = []

        self.handlers[queue_name].append(handler)
        logger.info(f"Registered handler for queue: {queue_name}")

    async def start_workers(self, worker_count: int = 1) -> None:
        """
        Start worker tasks for processing messages.

        Args:
            worker_count: Number of worker tasks per queue
        """
        self.is_running = True

        for queue_name, queue in self.queues.items():
            if queue_name not in self.workers:
                self.workers[queue_name] = []

            for i in range(worker_count):
                worker_task = asyncio.create_task(
                    self._worker_loop(queue_name, f"{queue_name}-worker-{i}")
                )
                self.workers[queue_name].append(worker_task)

        logger.info(f"Started {worker_count} workers for {len(self.queues)} queues")

    async def stop_workers(self) -> None:
        """Stop all worker tasks."""
        self.is_running = False

        all_tasks = []
        for workers in self.workers.values():
            all_tasks.extend(workers)

        if all_tasks:
            for task in all_tasks:
                task.cancel()

            await asyncio.gather(*all_tasks, return_exceptions=True)

        self.workers.clear()
        logger.info("Stopped all workers")

    async def _worker_loop(self, queue_name: str, worker_id: str) -> None:
        """
        Worker loop for processing messages from a specific queue.

        Args:
            queue_name: Name of the queue to process
            worker_id: Unique worker identifier
        """
        queue = self.queues[queue_name]
        handlers = self.handlers.get(queue_name, [])

        logger.info(f"Started worker {worker_id} for queue {queue_name}")

        while self.is_running:
            try:
                # Get next message
                message = await queue.dequeue(timeout=5)
                if not message:
                    continue

                logger.debug(f"Worker {worker_id} processing message {message.id}")

                # Find appropriate handler
                handler = None
                for h in handlers:
                    if h.can_handle(queue_name):
                        handler = h
                        break

                if not handler:
                    logger.error(f"No handler found for queue {queue_name}")
                    await queue.ack(message, success=False, error_message="No handler available")
                    continue

                # Process message
                try:
                    success = await handler.handle(message)
                    await queue.ack(message, success=success)

                    if success:
                        logger.debug(f"Worker {worker_id} successfully processed message {message.id}")
                    else:
                        logger.warning(f"Worker {worker_id} failed to process message {message.id}")

                except Exception as e:
                    logger.error(f"Worker {worker_id} error processing message {message.id}: {e}")
                    await queue.ack(message, success=False, error_message=str(e))

            except asyncio.CancelledError:
                logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} unexpected error: {e}")
                await asyncio.sleep(1)  # Brief pause before retrying

    async def publish(
        self,
        queue_name: str,
        payload: Dict[str, Any],
        priority: MessagePriority = MessagePriority.NORMAL,
        **kwargs
    ) -> str:
        """
        Publish a message to a queue.

        Args:
            queue_name: Name of the queue
            payload: Message payload
            priority: Message priority
            **kwargs: Additional message properties

        Returns:
            Message ID
        """
        if queue_name not in self.queues:
            raise ValueError(f"Queue {queue_name} does not exist")

        queue = self.queues[queue_name]
        return await queue.enqueue(payload, priority, **kwargs)

    async def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all queues."""
        stats = {
            "broker_status": "running" if self.is_running else "stopped",
            "queue_count": len(self.queues),
            "worker_count": sum(len(workers) for workers in self.workers.values()),
            "queues": {}
        }

        for queue_name, queue in self.queues.items():
            stats["queues"][queue_name] = await queue.get_stats()

        return stats