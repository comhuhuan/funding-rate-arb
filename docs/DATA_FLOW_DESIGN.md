# 数据流与消息传递机制设计

## 1. 数据流架构概览

本系统采用事件驱动架构，通过异步消息传递实现各模块间的解耦。数据流设计遵循单向数据流原则，确保数据一致性和系统可维护性。

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│  Data Pipeline  │───▶│  Data Consumers │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   - Exchanges   │    │ - Normalization │    │  - Strategies   │
│   - WebSocket   │    │ - Validation    │    │  - Risk Mgmt    │
│   - REST APIs   │    │ - Enrichment    │    │  - Order Mgmt   │
│   - Internal    │    │ - Aggregation   │    │  - Monitoring   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 2. 数据源与数据类型

### 2.1 数据源分类

#### 2.1.1 实时数据源
```python
class RealTimeDataSource:
    """实时数据源基类"""

    # 市场数据
    ORDERBOOK_STREAM = "orderbook"        # 订单簿数据流
    TRADE_STREAM = "trades"               # 成交数据流
    TICKER_STREAM = "ticker"              # 行情数据流

    # 账户数据
    ORDER_STREAM = "orders"               # 订单状态流
    POSITION_STREAM = "positions"         # 持仓数据流
    BALANCE_STREAM = "balance"            # 余额数据流

    # 资金费率
    FUNDING_RATE_STREAM = "funding_rate"  # 资金费率流
    FUNDING_SETTLEMENT = "funding_settle" # 资金费率结算流
```

#### 2.1.2 历史数据源
```python
class HistoricalDataSource:
    """历史数据源基类"""

    KLINE_DATA = "klines"                 # K线历史数据
    FUNDING_RATE_HISTORY = "funding_history"  # 资金费率历史
    TRADE_HISTORY = "trade_history"       # 交易历史
    PNL_HISTORY = "pnl_history"          # 盈亏历史
```

### 2.2 数据优先级与延迟要求

```python
from enum import Enum
from dataclasses import dataclass

class DataPriority(Enum):
    CRITICAL = 1    # 关键数据，<10ms
    HIGH = 2        # 高优先级，<50ms
    MEDIUM = 3      # 中等优先级，<200ms
    LOW = 4         # 低优先级，<1s

@dataclass
class DataStreamConfig:
    stream_type: str
    priority: DataPriority
    max_latency_ms: int
    buffer_size: int
    batch_size: int = 1
    retry_policy: Dict = None

# 数据流配置
DATA_STREAM_CONFIGS = {
    "orderbook": DataStreamConfig("orderbook", DataPriority.CRITICAL, 10, 1000),
    "trades": DataStreamConfig("trades", DataPriority.HIGH, 50, 5000),
    "funding_rate": DataStreamConfig("funding_rate", DataPriority.HIGH, 50, 1000),
    "orders": DataStreamConfig("orders", DataPriority.CRITICAL, 10, 1000),
    "positions": DataStreamConfig("positions", DataPriority.HIGH, 100, 500),
    "balance": DataStreamConfig("balance", DataPriority.MEDIUM, 200, 100),
}
```

## 3. 消息总线架构

### 3.1 消息总线设计

```python
import asyncio
from typing import Dict, List, Callable, Any
from abc import ABC, abstractmethod

class MessageBusInterface(ABC):
    """消息总线接口"""

    @abstractmethod
    async def publish(self, topic: str, message: Any, headers: Dict = None):
        """发布消息"""
        pass

    @abstractmethod
    async def subscribe(self, topic: str, handler: Callable, consumer_group: str = None):
        """订阅消息"""
        pass

    @abstractmethod
    async def unsubscribe(self, topic: str, handler: Callable):
        """取消订阅"""
        pass

class RedisStreamMessageBus(MessageBusInterface):
    """基于Redis Streams的消息总线"""

    def __init__(self, redis_client):
        self.redis = redis_client
        self.subscribers: Dict[str, List[Callable]] = {}
        self.consumer_groups: Dict[str, str] = {}

    async def publish(self, topic: str, message: Any, headers: Dict = None):
        """发布消息到Redis Stream"""
        data = {
            "payload": json.dumps(message),
            "timestamp": time.time(),
            "headers": json.dumps(headers or {})
        }
        await self.redis.xadd(topic, data)

    async def subscribe(self, topic: str, handler: Callable, consumer_group: str = None):
        """订阅Redis Stream消息"""
        if consumer_group:
            # 创建消费者组
            try:
                await self.redis.xgroup_create(topic, consumer_group, id='0', mkstream=True)
            except:
                pass  # 组可能已存在

            # 启动消费者
            asyncio.create_task(self._consume_group(topic, consumer_group, handler))
        else:
            if topic not in self.subscribers:
                self.subscribers[topic] = []
            self.subscribers[topic].append(handler)
            asyncio.create_task(self._consume_topic(topic))

    async def _consume_group(self, topic: str, group: str, handler: Callable):
        """消费者组消费"""
        consumer_id = f"consumer_{asyncio.current_task().get_name()}"

        while True:
            try:
                messages = await self.redis.xreadgroup(
                    group, consumer_id, {topic: '>'}, count=10, block=100
                )

                for stream, msgs in messages:
                    for msg_id, fields in msgs:
                        try:
                            payload = json.loads(fields[b'payload'])
                            headers = json.loads(fields.get(b'headers', b'{}'))
                            await handler(payload, headers)

                            # 确认消息处理完成
                            await self.redis.xack(topic, group, msg_id)
                        except Exception as e:
                            logger.error(f"Error processing message {msg_id}: {e}")
            except Exception as e:
                logger.error(f"Error in consumer group {group}: {e}")
                await asyncio.sleep(1)

    async def _consume_topic(self, topic: str):
        """主题消费"""
        last_id = '0-0'

        while True:
            try:
                messages = await self.redis.xread({topic: last_id}, count=10, block=100)

                for stream, msgs in messages:
                    for msg_id, fields in msgs:
                        try:
                            payload = json.loads(fields[b'payload'])
                            headers = json.loads(fields.get(b'headers', b'{}'))

                            # 并发处理所有订阅者
                            handlers = self.subscribers.get(topic, [])
                            await asyncio.gather(*[
                                handler(payload, headers) for handler in handlers
                            ])

                            last_id = msg_id
                        except Exception as e:
                            logger.error(f"Error processing message {msg_id}: {e}")
            except Exception as e:
                logger.error(f"Error consuming topic {topic}: {e}")
                await asyncio.sleep(1)
```

### 3.2 主题设计与路由规则

```python
class Topics:
    """消息主题定义"""

    # 市场数据主题
    MARKET_DATA_PREFIX = "market"
    ORDERBOOK = "market.orderbook"          # market.orderbook.{exchange}.{symbol}
    TRADES = "market.trades"                # market.trades.{exchange}.{symbol}
    TICKER = "market.ticker"                # market.ticker.{exchange}.{symbol}

    # 资金费率主题
    FUNDING_PREFIX = "funding"
    FUNDING_RATE = "funding.rate"           # funding.rate.{exchange}.{symbol}
    FUNDING_SETTLEMENT = "funding.settle"   # funding.settle.{exchange}.{symbol}
    FUNDING_PREDICTION = "funding.predict"  # funding.predict.{exchange}.{symbol}

    # 套利主题
    ARBITRAGE_PREFIX = "arbitrage"
    OPPORTUNITY_FOUND = "arbitrage.opportunity.found"
    OPPORTUNITY_EXPIRED = "arbitrage.opportunity.expired"
    OPPORTUNITY_EXECUTED = "arbitrage.opportunity.executed"

    # 交易主题
    TRADING_PREFIX = "trading"
    SIGNAL_GENERATED = "trading.signal.generated"
    ORDER_CREATED = "trading.order.created"
    ORDER_FILLED = "trading.order.filled"
    ORDER_CANCELLED = "trading.order.cancelled"
    ORDER_REJECTED = "trading.order.rejected"

    # 仓位主题
    POSITION_PREFIX = "position"
    POSITION_OPENED = "position.opened"
    POSITION_CLOSED = "position.closed"
    POSITION_UPDATED = "position.updated"

    # 风险主题
    RISK_PREFIX = "risk"
    RISK_ALERT = "risk.alert"
    RISK_LIMIT_BREACH = "risk.limit.breach"
    EMERGENCY_STOP = "risk.emergency.stop"

    # 系统主题
    SYSTEM_PREFIX = "system"
    HEALTH_CHECK = "system.health"
    CONFIG_UPDATE = "system.config.update"
    SHUTDOWN = "system.shutdown"

class TopicRouter:
    """主题路由器"""

    def __init__(self):
        self.routing_rules: Dict[str, List[str]] = {}

    def add_route(self, source_topic: str, target_topics: List[str]):
        """添加路由规则"""
        self.routing_rules[source_topic] = target_topics

    def route_message(self, topic: str, message: Any) -> List[str]:
        """根据路由规则确定目标主题"""
        targets = []

        # 精确匹配
        if topic in self.routing_rules:
            targets.extend(self.routing_rules[topic])

        # 模式匹配
        for pattern, target_list in self.routing_rules.items():
            if self._match_pattern(pattern, topic):
                targets.extend(target_list)

        return list(set(targets))  # 去重

    def _match_pattern(self, pattern: str, topic: str) -> bool:
        """模式匹配"""
        # 支持通配符 * 匹配
        import re
        pattern_regex = pattern.replace('*', '.*')
        return re.match(f"^{pattern_regex}$", topic) is not None
```

## 4. 实时数据处理流水线

### 4.1 数据处理流水线架构

```python
from typing import Protocol, TypeVar, Generic
import asyncio
from datetime import datetime

T = TypeVar('T')
U = TypeVar('U')

class DataProcessor(Protocol[T, U]):
    """数据处理器协议"""

    async def process(self, data: T) -> U:
        """处理数据"""
        pass

class Pipeline(Generic[T]):
    """数据处理流水线"""

    def __init__(self, name: str):
        self.name = name
        self.processors: List[DataProcessor] = []
        self.metrics = {}

    def add_processor(self, processor: DataProcessor) -> 'Pipeline':
        """添加处理器"""
        self.processors.append(processor)
        return self

    async def process(self, data: T) -> Any:
        """执行流水线处理"""
        start_time = time.time()
        result = data

        try:
            for i, processor in enumerate(self.processors):
                step_start = time.time()
                result = await processor.process(result)
                step_duration = time.time() - step_start

                # 记录每步处理时间
                self.metrics[f"step_{i}_duration"] = step_duration

            return result

        except Exception as e:
            logger.error(f"Pipeline {self.name} error at step {i}: {e}")
            raise
        finally:
            total_duration = time.time() - start_time
            self.metrics["total_duration"] = total_duration

class MarketDataPipeline:
    """市场数据处理流水线"""

    def __init__(self, message_bus: MessageBusInterface):
        self.message_bus = message_bus
        self.pipelines: Dict[str, Pipeline] = {}
        self._setup_pipelines()

    def _setup_pipelines(self):
        """设置处理流水线"""

        # 订单簿数据处理流水线
        orderbook_pipeline = Pipeline("orderbook") \
            .add_processor(DataValidator()) \
            .add_processor(DataNormalizer()) \
            .add_processor(OrderBookAggregator()) \
            .add_processor(MarketDataEnricher()) \
            .add_processor(DataCacheUpdater())

        self.pipelines["orderbook"] = orderbook_pipeline

        # 成交数据处理流水线
        trades_pipeline = Pipeline("trades") \
            .add_processor(DataValidator()) \
            .add_processor(DataNormalizer()) \
            .add_processor(TradeAggregator()) \
            .add_processor(VolumeCalculator()) \
            .add_processor(DataCacheUpdater())

        self.pipelines["trades"] = trades_pipeline

        # 资金费率处理流水线
        funding_pipeline = Pipeline("funding_rate") \
            .add_processor(DataValidator()) \
            .add_processor(FundingRateNormalizer()) \
            .add_processor(FundingRatePredictor()) \
            .add_processor(ArbitrageCalculator()) \
            .add_processor(DataCacheUpdater())

        self.pipelines["funding_rate"] = funding_pipeline

    async def process_data(self, data_type: str, raw_data: Dict) -> Any:
        """处理数据"""
        pipeline = self.pipelines.get(data_type)
        if not pipeline:
            raise ValueError(f"No pipeline found for data type: {data_type}")

        try:
            processed_data = await pipeline.process(raw_data)

            # 发布处理后的数据
            topic = f"{data_type}.processed"
            await self.message_bus.publish(topic, processed_data)

            return processed_data

        except Exception as e:
            logger.error(f"Error processing {data_type} data: {e}")
            raise

# 具体的数据处理器实现
class DataValidator:
    """数据验证器"""

    async def process(self, data: Dict) -> Dict:
        """验证数据格式和完整性"""
        required_fields = ['timestamp', 'exchange', 'symbol']

        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")

        # 验证时间戳合理性
        timestamp = data['timestamp']
        current_time = time.time() * 1000
        if abs(current_time - timestamp) > 60000:  # 1分钟内
            logger.warning(f"Data timestamp seems outdated: {timestamp}")

        return data

class DataNormalizer:
    """数据标准化器"""

    async def process(self, data: Dict) -> Dict:
        """标准化数据格式"""
        normalized = data.copy()

        # 标准化交易所名称
        normalized['exchange'] = normalized['exchange'].lower()

        # 标准化交易对格式
        symbol = normalized['symbol']
        if '/' not in symbol:
            # 处理没有分隔符的交易对
            if symbol.endswith('USDT'):
                base = symbol[:-4]
                quote = 'USDT'
                normalized['symbol'] = f"{base}/{quote}"

        # 确保价格和数量为Decimal类型
        if 'price' in normalized:
            normalized['price'] = Decimal(str(normalized['price']))

        if 'amount' in normalized:
            normalized['amount'] = Decimal(str(normalized['amount']))

        return normalized

class OrderBookAggregator:
    """订单簿聚合器"""

    def __init__(self):
        self.orderbooks: Dict[str, OrderBook] = {}

    async def process(self, data: Dict) -> OrderBook:
        """聚合订单簿数据"""
        key = f"{data['exchange']}:{data['symbol']}"

        # 更新订单簿
        orderbook = OrderBook(
            symbol=Symbol(
                base=data['symbol'].split('/')[0],
                quote=data['symbol'].split('/')[1],
                exchange=data['exchange'],
                symbol=data['symbol']
            ),
            bids=[Price(Decimal(str(p)), Decimal(str(v)), datetime.now())
                  for p, v in data['bids']],
            asks=[Price(Decimal(str(p)), Decimal(str(v)), datetime.now())
                  for p, v in data['asks']],
            timestamp=datetime.fromtimestamp(data['timestamp'] / 1000),
            exchange=data['exchange']
        )

        self.orderbooks[key] = orderbook
        return orderbook
```

### 4.2 背压处理与流量控制

```python
class BackpressureManager:
    """背压管理器"""

    def __init__(self, max_queue_size: int = 10000):
        self.max_queue_size = max_queue_size
        self.current_load: Dict[str, int] = {}
        self.drop_counters: Dict[str, int] = {}

    async def should_accept(self, stream_type: str) -> bool:
        """判断是否应该接收新数据"""
        current = self.current_load.get(stream_type, 0)

        if current >= self.max_queue_size:
            # 记录丢弃计数
            self.drop_counters[stream_type] = self.drop_counters.get(stream_type, 0) + 1

            # 采样丢弃策略
            if stream_type in ["orderbook", "trades"]:
                # 对于高频数据，采用采样策略
                return self.drop_counters[stream_type] % 10 == 0
            else:
                return False

        return True

    def update_load(self, stream_type: str, delta: int):
        """更新负载"""
        self.current_load[stream_type] = max(0, self.current_load.get(stream_type, 0) + delta)

class RateLimiter:
    """速率限制器"""

    def __init__(self, requests_per_second: int):
        self.rate = requests_per_second
        self.tokens = requests_per_second
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """获取令牌"""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update

            # 添加令牌
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True

            return False
```

## 5. 数据缓存与存储策略

### 5.1 多层缓存架构

```python
from abc import ABC, abstractmethod
from typing import Optional, Any

class CacheInterface(ABC):
    """缓存接口"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = None):
        """设置缓存"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """删除缓存"""
        pass

class TieredCache:
    """分层缓存"""

    def __init__(self):
        # L1: 内存缓存 (最快)
        self.l1_cache = MemoryCache(max_size=1000, ttl=10)

        # L2: Redis缓存 (中等速度)
        self.l2_cache = RedisCache(ttl=300)

        # L3: 数据库 (最慢但持久)
        self.l3_storage = DatabaseStorage()

    async def get(self, key: str) -> Optional[Any]:
        """分层获取数据"""
        # 先从L1获取
        value = await self.l1_cache.get(key)
        if value is not None:
            return value

        # 再从L2获取
        value = await self.l2_cache.get(key)
        if value is not None:
            # 回写L1
            await self.l1_cache.set(key, value)
            return value

        # 最后从L3获取
        value = await self.l3_storage.get(key)
        if value is not None:
            # 回写L2和L1
            await self.l2_cache.set(key, value)
            await self.l1_cache.set(key, value)
            return value

        return None

    async def set(self, key: str, value: Any):
        """设置所有层级的缓存"""
        await asyncio.gather(
            self.l1_cache.set(key, value),
            self.l2_cache.set(key, value),
            self.l3_storage.set(key, value)
        )

class DataCacheManager:
    """数据缓存管理器"""

    def __init__(self):
        self.cache = TieredCache()
        self.cache_policies: Dict[str, CachePolicy] = {
            "orderbook": CachePolicy(ttl=1, priority=Priority.HIGH),
            "trades": CachePolicy(ttl=5, priority=Priority.MEDIUM),
            "funding_rate": CachePolicy(ttl=60, priority=Priority.HIGH),
            "balance": CachePolicy(ttl=30, priority=Priority.MEDIUM),
        }

    async def cache_market_data(self, data_type: str, key: str, data: Any):
        """缓存市场数据"""
        policy = self.cache_policies.get(data_type)
        if policy:
            cache_key = f"{data_type}:{key}"
            await self.cache.set(cache_key, data)

    async def get_market_data(self, data_type: str, key: str) -> Optional[Any]:
        """获取市场数据"""
        cache_key = f"{data_type}:{key}"
        return await self.cache.get(cache_key)

@dataclass
class CachePolicy:
    ttl: int                    # 生存时间
    priority: Priority          # 优先级
    max_size: int = None       # 最大大小
    eviction_policy: str = "lru"  # 淘汰策略
```

### 5.2 数据持久化策略

```python
class DataPersistenceManager:
    """数据持久化管理器"""

    def __init__(self):
        self.time_series_db = InfluxDBClient()  # 时序数据
        self.relational_db = PostgreSQLClient()  # 关系数据
        self.document_db = MongoDBClient()        # 文档数据

    async def persist_market_data(self, data: OrderBook):
        """持久化市场数据"""
        # 写入时序数据库
        await self.time_series_db.write_points([
            {
                "measurement": "orderbook",
                "tags": {
                    "exchange": data.exchange,
                    "symbol": data.symbol.symbol,
                },
                "time": data.timestamp,
                "fields": {
                    "best_bid": float(data.bids[0].price) if data.bids else 0,
                    "best_ask": float(data.asks[0].price) if data.asks else 0,
                    "bid_volume": float(data.bids[0].volume) if data.bids else 0,
                    "ask_volume": float(data.asks[0].volume) if data.asks else 0,
                    "spread": float(data.asks[0].price - data.bids[0].price) if data.bids and data.asks else 0,
                }
            }
        ])

    async def persist_funding_rate(self, funding_rate: FundingRate):
        """持久化资金费率"""
        await self.time_series_db.write_points([
            {
                "measurement": "funding_rate",
                "tags": {
                    "exchange": funding_rate.exchange,
                    "symbol": funding_rate.symbol,
                },
                "time": funding_rate.timestamp,
                "fields": {
                    "rate": float(funding_rate.rate),
                    "predicted_rate": float(funding_rate.predicted_rate) if funding_rate.predicted_rate else None,
                    "next_funding_time": funding_rate.next_funding_time.timestamp(),
                }
            }
        ])

    async def persist_arbitrage_opportunity(self, opportunity: ArbitrageOpportunity):
        """持久化套利机会"""
        # 存储到文档数据库
        await self.document_db.insert_one("arbitrage_opportunities", {
            "id": opportunity.id,
            "type": opportunity.type.value,
            "symbol": opportunity.symbol,
            "exchanges": opportunity.exchanges,
            "expected_profit": float(opportunity.expected_profit),
            "risk_score": opportunity.risk_score,
            "created_at": opportunity.created_at,
            "valid_until": opportunity.valid_until,
            "metadata": opportunity.metadata
        })
```

## 6. 数据质量监控

### 6.1 数据质量检查

```python
class DataQualityMonitor:
    """数据质量监控器"""

    def __init__(self, message_bus: MessageBusInterface):
        self.message_bus = message_bus
        self.quality_metrics = {}
        self.alert_thresholds = {
            "latency_ms": 100,
            "missing_data_rate": 0.01,
            "data_accuracy": 0.95,
        }

    async def check_data_quality(self, data_type: str, data: Any) -> bool:
        """检查数据质量"""
        checks = [
            self._check_latency(data),
            self._check_completeness(data),
            self._check_consistency(data),
            self._check_accuracy(data_type, data),
        ]

        results = await asyncio.gather(*checks)
        quality_score = sum(results) / len(results)

        # 更新质量指标
        self.quality_metrics[data_type] = quality_score

        # 质量告警
        if quality_score < 0.8:
            await self._send_quality_alert(data_type, quality_score)

        return quality_score > 0.5

    async def _check_latency(self, data: Dict) -> float:
        """检查延迟"""
        if 'timestamp' not in data:
            return 0.0

        latency = time.time() * 1000 - data['timestamp']
        return 1.0 if latency < self.alert_thresholds["latency_ms"] else 0.0

    async def _check_completeness(self, data: Dict) -> float:
        """检查完整性"""
        required_fields = ['timestamp', 'exchange', 'symbol']
        missing_count = sum(1 for field in required_fields if field not in data)
        return 1.0 - (missing_count / len(required_fields))

    async def _check_consistency(self, data: Dict) -> float:
        """检查一致性"""
        # 检查价格是否合理
        if 'price' in data and data['price'] <= 0:
            return 0.0

        # 检查数量是否合理
        if 'amount' in data and data['amount'] <= 0:
            return 0.0

        return 1.0

    async def _send_quality_alert(self, data_type: str, quality_score: float):
        """发送质量告警"""
        alert = {
            "type": "data_quality_alert",
            "data_type": data_type,
            "quality_score": quality_score,
            "timestamp": datetime.now().isoformat(),
        }
        await self.message_bus.publish(Topics.SYSTEM_PREFIX + ".alert", alert)
```

## 7. 监控与观测

### 7.1 数据流监控

```python
class DataFlowMonitor:
    """数据流监控器"""

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.flow_stats: Dict[str, Dict] = {}

    async def track_data_flow(self, topic: str, data_size: int, processing_time: float):
        """跟踪数据流"""
        # 记录吞吐量
        self.metrics.record_counter(f"data.throughput.{topic}", 1)
        self.metrics.record_gauge(f"data.size.{topic}", data_size)
        self.metrics.record_timing(f"data.processing_time.{topic}", processing_time)

        # 更新统计信息
        if topic not in self.flow_stats:
            self.flow_stats[topic] = {
                "message_count": 0,
                "total_size": 0,
                "avg_processing_time": 0.0,
                "last_message_time": None,
            }

        stats = self.flow_stats[topic]
        stats["message_count"] += 1
        stats["total_size"] += data_size
        stats["avg_processing_time"] = (stats["avg_processing_time"] * (stats["message_count"] - 1) + processing_time) / stats["message_count"]
        stats["last_message_time"] = datetime.now()

    def get_flow_health(self) -> Dict[str, str]:
        """获取数据流健康状态"""
        health = {}
        now = datetime.now()

        for topic, stats in self.flow_stats.items():
            if not stats["last_message_time"]:
                health[topic] = "UNKNOWN"
                continue

            time_since_last = (now - stats["last_message_time"]).total_seconds()

            if time_since_last > 300:  # 5分钟没有数据
                health[topic] = "UNHEALTHY"
            elif time_since_last > 60:  # 1分钟没有数据
                health[topic] = "WARNING"
            else:
                health[topic] = "HEALTHY"

        return health

class AlertManager:
    """告警管理器"""

    def __init__(self, message_bus: MessageBusInterface):
        self.message_bus = message_bus
        self.alert_rules: List[AlertRule] = []
        self.alert_history: List[Alert] = []

    async def process_alert(self, alert_type: str, data: Dict):
        """处理告警"""
        alert = Alert(
            type=alert_type,
            severity=self._determine_severity(alert_type, data),
            message=self._format_message(alert_type, data),
            data=data,
            timestamp=datetime.now(),
        )

        # 检查是否需要发送告警
        if self._should_send_alert(alert):
            await self._send_alert(alert)
            self.alert_history.append(alert)

    def _determine_severity(self, alert_type: str, data: Dict) -> str:
        """确定告警级别"""
        severity_mapping = {
            "data_quality_alert": "WARNING",
            "latency_alert": "ERROR",
            "connection_lost": "CRITICAL",
            "risk_limit_breach": "CRITICAL",
        }
        return severity_mapping.get(alert_type, "INFO")

    async def _send_alert(self, alert: Alert):
        """发送告警"""
        # 发布到告警主题
        await self.message_bus.publish(Topics.RISK_ALERT, {
            "type": alert.type,
            "severity": alert.severity,
            "message": alert.message,
            "timestamp": alert.timestamp.isoformat(),
            "data": alert.data
        })

@dataclass
class Alert:
    type: str
    severity: str
    message: str
    data: Dict
    timestamp: datetime

@dataclass
class AlertRule:
    name: str
    condition: Callable
    action: Callable
    cooldown: int = 300  # 5分钟冷却期
```

这个数据流与消息传递机制设计提供了完整的实时数据处理框架，确保系统能够高效、可靠地处理大量的市场数据和交易信息，同时保持良好的可观测性和可维护性。