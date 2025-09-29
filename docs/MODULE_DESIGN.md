# 模块设计与接口规范

## 1. 模块划分概览

本系统采用分层架构和微服务设计模式，将整个系统划分为以下核心模块：

```
├── core/                          # 核心业务模块
│   ├── strategy/                  # 策略引擎
│   ├── arbitrage/                 # 套利计算引擎
│   ├── risk/                      # 风险管理
│   └── portfolio/                 # 组合管理
├── data/                          # 数据处理模块
│   ├── market/                    # 市场数据
│   ├── funding/                   # 资金费率
│   └── storage/                   # 数据存储
├── trading/                       # 交易执行模块
│   ├── order/                     # 订单管理
│   ├── execution/                 # 执行引擎
│   └── position/                  # 仓位管理
├── exchanges/                     # 交易所抽象层
│   ├── abstract/                  # 抽象接口
│   ├── ccxt_adapter/              # CCXT适配器
│   └── custom/                    # 自定义交易所实现
├── infrastructure/                # 基础设施
│   ├── cache/                     # 缓存服务
│   ├── messaging/                 # 消息队列
│   ├── config/                    # 配置管理
│   └── logging/                   # 日志服务
├── monitoring/                    # 监控模块
│   ├── metrics/                   # 指标收集
│   ├── alerts/                    # 告警服务
│   └── dashboard/                 # 监控面板
└── api/                          # API网关
    ├── rest/                     # REST API
    ├── websocket/                # WebSocket API
    └── grpc/                     # gRPC API
```

## 2. 数据模型定义

### 2.1 基础数据类型

```python
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum

class ExchangeType(Enum):
    BINANCE = "binance"
    OKX = "okx"
    BYBIT = "bybit"
    HUOBI = "huobi"
    CUSTOM = "custom"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    IOC = "ioc"
    FOK = "fok"

class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class ArbitrageType(Enum):
    SPOT_FUTURES = "spot_futures"
    FUTURES_FUTURES = "futures_futures"
    CROSS_EXCHANGE = "cross_exchange"
    FUNDING_RATE = "funding_rate"

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4
```

### 2.2 核心数据结构

```python
@dataclass
class Symbol:
    base: str          # 基础货币，如BTC
    quote: str         # 计价货币，如USDT
    exchange: str      # 交易所
    symbol: str        # 交易对符号，如BTC/USDT
    contract_type: Optional[str] = None  # 合约类型：spot, futures, perpetual

@dataclass
class Price:
    price: Decimal
    volume: Decimal
    timestamp: datetime

@dataclass
class OrderBook:
    symbol: Symbol
    bids: List[Price]  # 买盘深度
    asks: List[Price]  # 卖盘深度
    timestamp: datetime
    exchange: str

@dataclass
class Trade:
    id: str
    symbol: Symbol
    side: OrderSide
    amount: Decimal
    price: Decimal
    cost: Decimal
    fee: Decimal
    timestamp: datetime
    exchange: str

@dataclass
class FundingRate:
    exchange: str
    symbol: str
    rate: Decimal                    # 资金费率
    timestamp: datetime              # 当前时间
    next_funding_time: datetime      # 下次结算时间
    funding_interval: int            # 结算间隔（秒）
    predicted_rate: Optional[Decimal] = None  # 预测费率

@dataclass
class Position:
    symbol: Symbol
    side: OrderSide
    amount: Decimal              # 持仓数量
    entry_price: Decimal         # 开仓均价
    mark_price: Decimal          # 标记价格
    unrealized_pnl: Decimal      # 未实现盈亏
    margin: Decimal              # 保证金
    leverage: float              # 杠杆倍数
    timestamp: datetime

@dataclass
class Balance:
    currency: str
    free: Decimal                # 可用余额
    used: Decimal                # 冻结余额
    total: Decimal               # 总余额
    timestamp: datetime

@dataclass
class Order:
    id: str
    client_order_id: str
    symbol: Symbol
    side: OrderSide
    order_type: OrderType
    amount: Decimal
    price: Optional[Decimal]
    filled: Decimal
    remaining: Decimal
    status: OrderStatus
    timestamp: datetime
    fee: Optional[Decimal] = None
    average_price: Optional[Decimal] = None

@dataclass
class ArbitrageOpportunity:
    id: str
    type: ArbitrageType
    symbol: str
    exchanges: List[str]
    entry_signals: List['TradingSignal']
    expected_profit: Decimal
    max_profit: Decimal
    risk_score: float
    max_position_size: Decimal
    liquidity_score: float
    urgency: Priority
    valid_until: datetime
    created_at: datetime
    metadata: Dict = None

@dataclass
class TradingSignal:
    id: str
    strategy_id: str
    symbol: Symbol
    side: OrderSide
    signal_type: str             # entry, exit, close
    price: Optional[Decimal]
    amount: Decimal
    confidence: float            # 信号置信度 0-1
    urgency: Priority
    valid_until: datetime
    created_at: datetime
    metadata: Dict = None
```

## 3. 核心模块接口设计

### 3.1 交易所抽象层接口

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, AsyncIterator

class ExchangeInterface(ABC):
    """交易所统一接口"""

    @abstractmethod
    async def get_symbols(self) -> List[Symbol]:
        """获取所有交易对"""
        pass

    @abstractmethod
    async def get_orderbook(self, symbol: str, limit: int = 100) -> OrderBook:
        """获取订单簿"""
        pass

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict:
        """获取行情数据"""
        pass

    @abstractmethod
    async def get_funding_rate(self, symbol: str) -> FundingRate:
        """获取资金费率"""
        pass

    @abstractmethod
    async def get_funding_rate_history(self, symbol: str, since: datetime, limit: int = 100) -> List[FundingRate]:
        """获取历史资金费率"""
        pass

    @abstractmethod
    async def get_balance(self) -> Dict[str, Balance]:
        """获取账户余额"""
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """获取持仓信息"""
        pass

    @abstractmethod
    async def create_order(self, order: Order) -> Order:
        """创建订单"""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """取消订单"""
        pass

    @abstractmethod
    async def get_order(self, order_id: str, symbol: str) -> Order:
        """查询订单"""
        pass

    @abstractmethod
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """获取未成交订单"""
        pass

    @abstractmethod
    async def subscribe_orderbook(self, symbol: str) -> AsyncIterator[OrderBook]:
        """订阅订单簿推送"""
        pass

    @abstractmethod
    async def subscribe_trades(self, symbol: str) -> AsyncIterator[Trade]:
        """订阅成交数据推送"""
        pass

    @abstractmethod
    async def subscribe_funding_rate(self, symbol: str) -> AsyncIterator[FundingRate]:
        """订阅资金费率推送"""
        pass

class ExchangeManager:
    """交易所管理器"""

    def __init__(self):
        self._exchanges: Dict[str, ExchangeInterface] = {}

    def register_exchange(self, name: str, exchange: ExchangeInterface):
        """注册交易所"""
        self._exchanges[name] = exchange

    def get_exchange(self, name: str) -> ExchangeInterface:
        """获取交易所实例"""
        return self._exchanges.get(name)

    def get_all_exchanges(self) -> Dict[str, ExchangeInterface]:
        """获取所有交易所"""
        return self._exchanges

    async def get_funding_rates_all(self, symbol: str) -> Dict[str, FundingRate]:
        """获取所有交易所的资金费率"""
        rates = {}
        for name, exchange in self._exchanges.items():
            try:
                rate = await exchange.get_funding_rate(symbol)
                rates[name] = rate
            except Exception as e:
                # 记录错误但不中断其他交易所的查询
                pass
        return rates
```

### 3.2 市场数据模块接口

```python
class MarketDataService:
    """市场数据服务"""

    async def subscribe_symbol(self, exchange: str, symbol: str):
        """订阅交易对数据"""
        pass

    async def unsubscribe_symbol(self, exchange: str, symbol: str):
        """取消订阅交易对数据"""
        pass

    async def get_latest_orderbook(self, exchange: str, symbol: str) -> Optional[OrderBook]:
        """获取最新订单簿"""
        pass

    async def get_latest_price(self, exchange: str, symbol: str) -> Optional[Decimal]:
        """获取最新价格"""
        pass

    def on_orderbook_update(self, callback):
        """注册订单簿更新回调"""
        pass

    def on_trade_update(self, callback):
        """注册成交数据更新回调"""
        pass

class FundingRateService:
    """资金费率服务"""

    async def get_funding_rate(self, exchange: str, symbol: str) -> Optional[FundingRate]:
        """获取资金费率"""
        pass

    async def get_all_funding_rates(self, symbol: str) -> Dict[str, FundingRate]:
        """获取所有交易所的资金费率"""
        pass

    async def get_funding_rate_spread(self, symbol: str) -> Dict:
        """获取资金费率价差"""
        pass

    async def predict_funding_rate(self, exchange: str, symbol: str) -> Optional[Decimal]:
        """预测下期资金费率"""
        pass

    def on_funding_rate_update(self, callback):
        """注册资金费率更新回调"""
        pass

    def on_funding_settlement(self, callback):
        """注册资金费率结算回调"""
        pass
```

### 3.3 套利计算引擎接口

```python
class ArbitrageCalculator:
    """套利计算器"""

    def __init__(self, config: Dict):
        self.config = config

    async def calculate_spot_futures_arbitrage(self, symbol: str) -> List[ArbitrageOpportunity]:
        """计算现货期货套利机会"""
        pass

    async def calculate_futures_arbitrage(self, symbol: str) -> List[ArbitrageOpportunity]:
        """计算期货套利机会"""
        pass

    async def calculate_cross_exchange_arbitrage(self, symbol: str) -> List[ArbitrageOpportunity]:
        """计算跨交易所套利机会"""
        pass

    async def calculate_funding_rate_arbitrage(self, symbol: str) -> List[ArbitrageOpportunity]:
        """计算资金费率套利机会"""
        pass

    def calculate_profit(self, opportunity: ArbitrageOpportunity, position_size: Decimal) -> Decimal:
        """计算套利利润"""
        pass

    def calculate_risk(self, opportunity: ArbitrageOpportunity) -> float:
        """计算套利风险"""
        pass

    def estimate_slippage(self, exchange: str, symbol: str, amount: Decimal) -> Decimal:
        """估算滑点"""
        pass

class ArbitrageEngine:
    """套利引擎"""

    def __init__(self, calculator: ArbitrageCalculator):
        self.calculator = calculator
        self.opportunities: Dict[str, ArbitrageOpportunity] = {}

    async def scan_opportunities(self) -> List[ArbitrageOpportunity]:
        """扫描套利机会"""
        pass

    async def rank_opportunities(self, opportunities: List[ArbitrageOpportunity]) -> List[ArbitrageOpportunity]:
        """对套利机会进行排序"""
        pass

    async def filter_opportunities(self, opportunities: List[ArbitrageOpportunity]) -> List[ArbitrageOpportunity]:
        """过滤套利机会"""
        pass

    def get_top_opportunities(self, n: int = 10) -> List[ArbitrageOpportunity]:
        """获取TOP N套利机会"""
        pass

    def on_opportunity_found(self, callback):
        """注册套利机会发现回调"""
        pass

    def on_opportunity_expired(self, callback):
        """注册套利机会过期回调"""
        pass
```

### 3.4 策略引擎接口

```python
class StrategyBase(ABC):
    """策略基类"""

    def __init__(self, config: Dict):
        self.config = config
        self.is_running = False

    @abstractmethod
    async def initialize(self):
        """初始化策略"""
        pass

    @abstractmethod
    async def on_market_data(self, data: OrderBook):
        """处理市场数据"""
        pass

    @abstractmethod
    async def on_funding_rate_update(self, funding_rate: FundingRate):
        """处理资金费率更新"""
        pass

    @abstractmethod
    async def on_arbitrage_opportunity(self, opportunity: ArbitrageOpportunity):
        """处理套利机会"""
        pass

    @abstractmethod
    async def generate_signals(self) -> List[TradingSignal]:
        """生成交易信号"""
        pass

    @abstractmethod
    def calculate_position_size(self, signal: TradingSignal) -> Decimal:
        """计算仓位大小"""
        pass

    async def start(self):
        """启动策略"""
        self.is_running = True
        await self.initialize()

    async def stop(self):
        """停止策略"""
        self.is_running = False

class StrategyManager:
    """策略管理器"""

    def __init__(self):
        self.strategies: Dict[str, StrategyBase] = {}

    def register_strategy(self, name: str, strategy: StrategyBase):
        """注册策略"""
        self.strategies[name] = strategy

    async def start_strategy(self, name: str):
        """启动策略"""
        if name in self.strategies:
            await self.strategies[name].start()

    async def stop_strategy(self, name: str):
        """停止策略"""
        if name in self.strategies:
            await self.strategies[name].stop()

    async def start_all(self):
        """启动所有策略"""
        for strategy in self.strategies.values():
            await strategy.start()

    async def stop_all(self):
        """停止所有策略"""
        for strategy in self.strategies.values():
            await strategy.stop()
```

### 3.5 订单管理系统接口

```python
class OrderManagerInterface(ABC):
    """订单管理接口"""

    @abstractmethod
    async def create_order(self, signal: TradingSignal) -> Order:
        """根据交易信号创建订单"""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        pass

    @abstractmethod
    async def modify_order(self, order_id: str, **kwargs) -> Order:
        """修改订单"""
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """获取订单状态"""
        pass

    @abstractmethod
    async def get_open_orders(self) -> List[Order]:
        """获取未成交订单"""
        pass

class OrderExecutor:
    """订单执行器"""

    def __init__(self, exchange_manager: ExchangeManager):
        self.exchange_manager = exchange_manager
        self.pending_orders: Dict[str, Order] = {}

    async def execute_order(self, order: Order) -> Order:
        """执行订单"""
        pass

    async def execute_batch_orders(self, orders: List[Order]) -> List[Order]:
        """批量执行订单"""
        pass

    async def smart_order_routing(self, signal: TradingSignal) -> List[Order]:
        """智能订单路由"""
        pass

    def on_order_filled(self, callback):
        """注册订单成交回调"""
        pass

    def on_order_cancelled(self, callback):
        """注册订单取消回调"""
        pass

class PositionManager:
    """仓位管理器"""

    async def get_positions(self, exchange: Optional[str] = None) -> List[Position]:
        """获取持仓"""
        pass

    async def get_net_position(self, symbol: str) -> Decimal:
        """获取净持仓"""
        pass

    async def close_position(self, exchange: str, symbol: str, amount: Optional[Decimal] = None):
        """平仓"""
        pass

    async def hedge_position(self, position: Position):
        """对冲仓位"""
        pass

    def calculate_exposure(self) -> Dict[str, Decimal]:
        """计算风险敞口"""
        pass
```

### 3.6 风险管理模块接口

```python
class RiskManagerInterface(ABC):
    """风险管理接口"""

    @abstractmethod
    async def check_pre_trade_risk(self, signal: TradingSignal) -> bool:
        """交易前风险检查"""
        pass

    @abstractmethod
    async def check_position_risk(self, position: Position) -> bool:
        """持仓风险检查"""
        pass

    @abstractmethod
    async def check_portfolio_risk(self) -> Dict[str, float]:
        """组合风险检查"""
        pass

    @abstractmethod
    async def calculate_var(self, confidence: float = 0.95) -> Decimal:
        """计算风险价值VaR"""
        pass

    @abstractmethod
    async def emergency_stop(self, reason: str):
        """紧急停止"""
        pass

class RiskLimits:
    """风险限制配置"""

    max_position_size: Dict[str, Decimal]      # 最大持仓
    max_daily_loss: Decimal                    # 最大日损失
    max_drawdown: float                        # 最大回撤
    var_limit: Decimal                         # VaR限制
    correlation_limit: float                   # 相关性限制
    leverage_limit: float                      # 杠杆限制

class RiskMonitor:
    """风险监控"""

    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self.alerts = []

    async def monitor_real_time(self):
        """实时风险监控"""
        pass

    async def check_limits(self) -> List[str]:
        """检查风险限制"""
        pass

    def generate_risk_report(self) -> Dict:
        """生成风险报告"""
        pass

    def on_risk_alert(self, callback):
        """注册风险告警回调"""
        pass
```

## 4. 消息传递与事件系统

### 4.1 事件定义

```python
from dataclasses import dataclass
from typing import Any
from datetime import datetime

@dataclass
class Event:
    event_type: str
    data: Any
    timestamp: datetime
    source: str
    correlation_id: Optional[str] = None

class EventType:
    # 市场数据事件
    ORDERBOOK_UPDATE = "orderbook_update"
    TRADE_UPDATE = "trade_update"
    PRICE_UPDATE = "price_update"

    # 资金费率事件
    FUNDING_RATE_UPDATE = "funding_rate_update"
    FUNDING_SETTLEMENT = "funding_settlement"

    # 套利事件
    ARBITRAGE_OPPORTUNITY_FOUND = "arbitrage_opportunity_found"
    ARBITRAGE_OPPORTUNITY_EXPIRED = "arbitrage_opportunity_expired"

    # 交易事件
    ORDER_CREATED = "order_created"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    ORDER_REJECTED = "order_rejected"

    # 仓位事件
    POSITION_OPENED = "position_opened"
    POSITION_CLOSED = "position_closed"
    POSITION_UPDATED = "position_updated"

    # 风险事件
    RISK_ALERT = "risk_alert"
    RISK_LIMIT_BREACH = "risk_limit_breach"
    EMERGENCY_STOP = "emergency_stop"

    # 系统事件
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    EXCHANGE_CONNECTED = "exchange_connected"
    EXCHANGE_DISCONNECTED = "exchange_disconnected"
```

### 4.2 消息总线接口

```python
class MessageBus(ABC):
    """消息总线接口"""

    @abstractmethod
    async def publish(self, topic: str, event: Event):
        """发布事件"""
        pass

    @abstractmethod
    async def subscribe(self, topic: str, handler):
        """订阅主题"""
        pass

    @abstractmethod
    async def unsubscribe(self, topic: str, handler):
        """取消订阅"""
        pass

class EventHandler(ABC):
    """事件处理器基类"""

    @abstractmethod
    async def handle(self, event: Event):
        """处理事件"""
        pass

class EventRouter:
    """事件路由器"""

    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.handlers: Dict[str, List[EventHandler]] = {}

    def register_handler(self, event_type: str, handler: EventHandler):
        """注册事件处理器"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)

    async def route_event(self, event: Event):
        """路由事件"""
        handlers = self.handlers.get(event.event_type, [])
        for handler in handlers:
            await handler.handle(event)
```

## 5. 配置管理

### 5.1 配置模型

```python
@dataclass
class ExchangeConfig:
    name: str
    api_key: str
    secret_key: str
    passphrase: Optional[str] = None
    sandbox: bool = False
    rate_limit: int = 1000
    timeout: int = 30
    custom_settings: Dict = None

@dataclass
class StrategyConfig:
    name: str
    enabled: bool
    symbols: List[str]
    parameters: Dict
    risk_limits: Dict
    position_size: Decimal
    max_positions: int

@dataclass
class RiskConfig:
    max_position_size: Dict[str, Decimal]
    max_daily_loss: Decimal
    max_drawdown: float
    var_limit: Decimal
    stop_loss_threshold: float
    take_profit_threshold: float

@dataclass
class SystemConfig:
    exchanges: List[ExchangeConfig]
    strategies: List[StrategyConfig]
    risk: RiskConfig
    database_url: str
    redis_url: str
    log_level: str
    monitoring_enabled: bool
```

### 5.2 配置管理器

```python
class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config: SystemConfig = None

    def load_config(self) -> SystemConfig:
        """加载配置"""
        pass

    def save_config(self, config: SystemConfig):
        """保存配置"""
        pass

    def reload_config(self):
        """重新加载配置"""
        pass

    def get_exchange_config(self, name: str) -> Optional[ExchangeConfig]:
        """获取交易所配置"""
        pass

    def get_strategy_config(self, name: str) -> Optional[StrategyConfig]:
        """获取策略配置"""
        pass

    def update_config(self, updates: Dict):
        """更新配置"""
        pass

    def validate_config(self, config: SystemConfig) -> bool:
        """验证配置"""
        pass
```

## 6. 监控与指标

### 6.1 指标定义

```python
class Metrics:
    """系统指标"""

    # 性能指标
    latency_orderbook_update = "latency.orderbook_update"
    latency_order_execution = "latency.order_execution"
    latency_arbitrage_detection = "latency.arbitrage_detection"

    # 业务指标
    arbitrage_opportunities_found = "arbitrage.opportunities_found"
    arbitrage_opportunities_executed = "arbitrage.opportunities_executed"
    total_profit = "trading.total_profit"
    total_trades = "trading.total_trades"
    win_rate = "trading.win_rate"

    # 系统指标
    exchange_connections = "system.exchange_connections"
    active_strategies = "system.active_strategies"
    cpu_usage = "system.cpu_usage"
    memory_usage = "system.memory_usage"

    # 风险指标
    current_var = "risk.current_var"
    max_drawdown = "risk.max_drawdown"
    position_exposure = "risk.position_exposure"

class MetricsCollector:
    """指标收集器"""

    def record_counter(self, metric: str, value: int = 1, tags: Dict = None):
        """记录计数器指标"""
        pass

    def record_gauge(self, metric: str, value: float, tags: Dict = None):
        """记录仪表指标"""
        pass

    def record_histogram(self, metric: str, value: float, tags: Dict = None):
        """记录直方图指标"""
        pass

    def record_timing(self, metric: str, duration: float, tags: Dict = None):
        """记录时间指标"""
        pass
```

这个模块设计文档提供了系统各个组件的详细接口规范和数据模型定义，为后续的开发实现奠定了坚实的基础。每个模块都是高度解耦的，可以独立开发、测试和部署。