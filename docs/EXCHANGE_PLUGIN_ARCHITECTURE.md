# 交易所插件化架构设计

## 1. 插件化架构概览

交易所插件化架构是本系统的核心基础设施，提供统一的交易所接入接口，支持热插拔、动态加载、配置化管理等特性。架构设计优先使用CCXT库，同时为不支持的交易所提供自定义扩展能力。

```
┌─────────────────────────────────────────────────────────────┐
│                   交易所插件化架构                            │
├─────────────────────────────────────────────────────────────┤
│   Plugin Manager   │   Registry   │   Loader   │   Config   │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                    统一接口层                                │
├─────────────────────────────────────────────────────────────┤
│  Market Data  │  Trading  │  Account  │  WebSocket  │  Auth  │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                    适配器层                                  │
├─────────────────────────────────────────────────────────────┤
│  CCXT Adapter  │  Custom Adapter  │  Proxy Adapter         │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                   交易所实现层                               │
├─────────────────────────────────────────────────────────────┤
│  Binance  │  OKX  │  Bybit  │  Huobi  │  Custom  │  Future │
└─────────────────────────────────────────────────────────────┘
```

## 2. 插件接口定义

### 2.1 核心插件接口

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, AsyncIterator, Any, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio

class PluginType(Enum):
    CCXT = "ccxt"
    CUSTOM = "custom"
    PROXY = "proxy"
    AGGREGATED = "aggregated"

class ConnectionStatus(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"

@dataclass
class ExchangeCapabilities:
    """交易所能力定义"""

    # 交易功能
    spot_trading: bool = False
    futures_trading: bool = False
    margin_trading: bool = False
    options_trading: bool = False

    # 数据功能
    orderbook_stream: bool = False
    trades_stream: bool = False
    ticker_stream: bool = False
    kline_stream: bool = False

    # 账户功能
    balance_query: bool = False
    position_query: bool = False
    order_history: bool = False
    trade_history: bool = False

    # 特殊功能
    funding_rate: bool = False
    lending: bool = False
    staking: bool = False

@dataclass
class ExchangeMetadata:
    """交易所元数据"""

    name: str
    display_name: str
    version: str
    plugin_type: PluginType
    capabilities: ExchangeCapabilities
    rate_limits: Dict[str, int]
    supported_symbols: List[str]
    base_currencies: List[str]
    quote_currencies: List[str]
    min_trade_amounts: Dict[str, float]
    fee_structure: Dict[str, float]
    requires_credentials: bool = True

class ExchangePluginInterface(ABC):
    """交易所插件接口"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metadata: ExchangeMetadata = None
        self.connection_status = ConnectionStatus.DISCONNECTED
        self.rate_limiter: Optional[RateLimiter] = None
        self.event_handlers: Dict[str, List[Callable]] = {}

    @abstractmethod
    async def initialize(self) -> bool:
        """初始化插件"""
        pass

    @abstractmethod
    async def connect(self) -> bool:
        """建立连接"""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """断开连接"""
        pass

    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        pass

    @abstractmethod
    def get_metadata(self) -> ExchangeMetadata:
        """获取交易所元数据"""
        pass

    # 市场数据接口
    @abstractmethod
    async def get_symbols(self) -> List[Symbol]:
        """获取交易对列表"""
        pass

    @abstractmethod
    async def get_orderbook(self, symbol: str, limit: int = 100) -> OrderBook:
        """获取订单簿"""
        pass

    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """获取行情数据"""
        pass

    @abstractmethod
    async def get_trades(self, symbol: str, limit: int = 100) -> List[Trade]:
        """获取最近成交"""
        pass

    @abstractmethod
    async def get_klines(self, symbol: str, timeframe: str,
                        since: Optional[int] = None, limit: int = 100) -> List[Dict]:
        """获取K线数据"""
        pass

    # 资金费率接口
    @abstractmethod
    async def get_funding_rate(self, symbol: str) -> Optional[FundingRate]:
        """获取资金费率"""
        pass

    @abstractmethod
    async def get_funding_rate_history(self, symbol: str,
                                     since: Optional[int] = None,
                                     limit: int = 100) -> List[FundingRate]:
        """获取历史资金费率"""
        pass

    # 账户接口
    @abstractmethod
    async def get_balance(self) -> Dict[str, Balance]:
        """获取账户余额"""
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """获取持仓信息"""
        pass

    # 交易接口
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
    async def get_order_history(self, symbol: Optional[str] = None,
                              since: Optional[int] = None,
                              limit: int = 100) -> List[Order]:
        """获取订单历史"""
        pass

    # WebSocket接口
    @abstractmethod
    async def subscribe_orderbook(self, symbol: str) -> AsyncIterator[OrderBook]:
        """订阅订单簿"""
        pass

    @abstractmethod
    async def subscribe_trades(self, symbol: str) -> AsyncIterator[Trade]:
        """订阅成交数据"""
        pass

    @abstractmethod
    async def subscribe_ticker(self, symbol: str) -> AsyncIterator[Dict]:
        """订阅行情数据"""
        pass

    @abstractmethod
    async def subscribe_funding_rate(self, symbol: str) -> AsyncIterator[FundingRate]:
        """订阅资金费率"""
        pass

    @abstractmethod
    async def subscribe_user_data(self) -> AsyncIterator[Dict]:
        """订阅用户数据流"""
        pass

    # 事件处理
    def register_event_handler(self, event: str, handler: Callable):
        """注册事件处理器"""
        if event not in self.event_handlers:
            self.event_handlers[event] = []
        self.event_handlers[event].append(handler)

    async def emit_event(self, event: str, data: Any):
        """触发事件"""
        handlers = self.event_handlers.get(event, [])
        for handler in handlers:
            try:
                await handler(data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")
```

### 2.2 CCXT适配器实现

```python
import ccxt.async_support as ccxt
from typing import Dict, Any

class CCXTAdapter(ExchangePluginInterface):
    """CCXT交易所适配器"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.exchange_class = getattr(ccxt, config['exchange_id'])
        self.exchange = None
        self._setup_metadata()

    def _setup_metadata(self):
        """设置元数据"""
        exchange_info = self.exchange_class()

        # 映射CCXT能力到我们的能力定义
        capabilities = ExchangeCapabilities(
            spot_trading=exchange_info.has.get('spot', False),
            futures_trading=exchange_info.has.get('future', False),
            margin_trading=exchange_info.has.get('margin', False),
            orderbook_stream=exchange_info.has.get('ws', False),
            trades_stream=exchange_info.has.get('ws', False),
            funding_rate=exchange_info.has.get('fetchFundingRate', False),
        )

        self.metadata = ExchangeMetadata(
            name=self.config['exchange_id'],
            display_name=exchange_info.name,
            version="1.0.0",
            plugin_type=PluginType.CCXT,
            capabilities=capabilities,
            rate_limits=exchange_info.rateLimit,
            supported_symbols=[],  # 需要连接后获取
            base_currencies=[],
            quote_currencies=[],
            min_trade_amounts={},
            fee_structure={},
        )

    async def initialize(self) -> bool:
        """初始化CCXT交易所"""
        try:
            # 创建交易所实例
            self.exchange = self.exchange_class({
                'apiKey': self.config.get('api_key', ''),
                'secret': self.config.get('secret_key', ''),
                'password': self.config.get('passphrase', ''),
                'sandbox': self.config.get('sandbox', False),
                'enableRateLimit': True,
                'timeout': self.config.get('timeout', 30000),
                **self.config.get('extra_params', {})
            })

            # 设置速率限制器
            self.rate_limiter = RateLimiter(self.exchange.rateLimit / 1000)

            return True

        except Exception as e:
            logger.error(f"CCXT initialization failed: {e}")
            return False

    async def connect(self) -> bool:
        """建立连接"""
        try:
            self.connection_status = ConnectionStatus.CONNECTING

            # 测试连接
            await self.exchange.load_markets()

            # 更新元数据
            await self._update_metadata()

            self.connection_status = ConnectionStatus.CONNECTED
            await self.emit_event('connected', {'exchange': self.metadata.name})

            return True

        except Exception as e:
            logger.error(f"CCXT connection failed: {e}")
            self.connection_status = ConnectionStatus.ERROR
            return False

    async def disconnect(self) -> bool:
        """断开连接"""
        try:
            if self.exchange:
                await self.exchange.close()

            self.connection_status = ConnectionStatus.DISCONNECTED
            await self.emit_event('disconnected', {'exchange': self.metadata.name})

            return True

        except Exception as e:
            logger.error(f"CCXT disconnection failed: {e}")
            return False

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            # 测试API连通性
            start_time = time.time()
            await self.exchange.fetch_status()
            latency = (time.time() - start_time) * 1000

            return {
                'status': 'healthy',
                'latency_ms': latency,
                'connection_status': self.connection_status.value,
                'last_check': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'connection_status': self.connection_status.value,
                'last_check': datetime.now().isoformat()
            }

    async def _update_metadata(self):
        """更新元数据"""
        markets = self.exchange.markets

        self.metadata.supported_symbols = list(markets.keys())
        self.metadata.base_currencies = list(set(m['base'] for m in markets.values()))
        self.metadata.quote_currencies = list(set(m['quote'] for m in markets.values()))

        # 更新最小交易量
        for symbol, market in markets.items():
            if market.get('limits', {}).get('amount', {}).get('min'):
                self.metadata.min_trade_amounts[symbol] = market['limits']['amount']['min']

    def get_metadata(self) -> ExchangeMetadata:
        """获取元数据"""
        return self.metadata

    # 实现具体的API方法
    async def get_symbols(self) -> List[Symbol]:
        """获取交易对列表"""
        await self.rate_limiter.acquire()

        markets = await self.exchange.fetch_markets()
        symbols = []

        for market in markets:
            symbols.append(Symbol(
                base=market['base'],
                quote=market['quote'],
                exchange=self.metadata.name,
                symbol=market['symbol'],
                contract_type=market.get('type', 'spot')
            ))

        return symbols

    async def get_orderbook(self, symbol: str, limit: int = 100) -> OrderBook:
        """获取订单簿"""
        await self.rate_limiter.acquire()

        orderbook_data = await self.exchange.fetch_order_book(symbol, limit)

        bids = [Price(Decimal(str(p)), Decimal(str(v)), datetime.now())
                for p, v in orderbook_data['bids']]
        asks = [Price(Decimal(str(p)), Decimal(str(v)), datetime.now())
                for p, v in orderbook_data['asks']]

        return OrderBook(
            symbol=Symbol.from_string(symbol, self.metadata.name),
            bids=bids,
            asks=asks,
            timestamp=datetime.fromtimestamp(orderbook_data['timestamp'] / 1000),
            exchange=self.metadata.name
        )

    async def get_funding_rate(self, symbol: str) -> Optional[FundingRate]:
        """获取资金费率"""
        if not self.metadata.capabilities.funding_rate:
            return None

        await self.rate_limiter.acquire()

        try:
            funding_data = await self.exchange.fetch_funding_rate(symbol)

            return FundingRate(
                exchange=self.metadata.name,
                symbol=symbol,
                rate=Decimal(str(funding_data['fundingRate'])),
                timestamp=datetime.fromtimestamp(funding_data['timestamp'] / 1000),
                next_funding_time=datetime.fromtimestamp(funding_data['fundingDatetime'] / 1000),
                funding_interval=8 * 3600  # 8小时，具体值需要从交易所获取
            )

        except Exception as e:
            logger.error(f"Failed to fetch funding rate for {symbol}: {e}")
            return None

    async def create_order(self, order: Order) -> Order:
        """创建订单"""
        await self.rate_limiter.acquire()

        # 转换订单类型
        ccxt_type = self._convert_order_type(order.order_type)
        ccxt_side = order.side.value

        try:
            result = await self.exchange.create_order(
                symbol=order.symbol.symbol,
                type=ccxt_type,
                side=ccxt_side,
                amount=float(order.amount),
                price=float(order.price) if order.price else None,
                params={}
            )

            # 更新订单信息
            order.id = result['id']
            order.status = self._convert_order_status(result['status'])
            order.filled = Decimal(str(result.get('filled', 0)))
            order.remaining = Decimal(str(result.get('remaining', order.amount)))

            return order

        except Exception as e:
            logger.error(f"Order creation failed: {e}")
            raise

    def _convert_order_type(self, order_type: OrderType) -> str:
        """转换订单类型"""
        mapping = {
            OrderType.MARKET: 'market',
            OrderType.LIMIT: 'limit',
            OrderType.IOC: 'limit',  # CCXT中可能需要特殊处理
            OrderType.FOK: 'limit',
        }
        return mapping.get(order_type, 'limit')

    def _convert_order_status(self, ccxt_status: str) -> str:
        """转换订单状态"""
        mapping = {
            'open': OrderState.OPEN.value,
            'closed': OrderState.FILLED.value,
            'canceled': OrderState.CANCELLED.value,
            'cancelled': OrderState.CANCELLED.value,
            'rejected': OrderState.REJECTED.value,
        }
        return mapping.get(ccxt_status, OrderState.UNKNOWN.value)

    # WebSocket订阅方法
    async def subscribe_orderbook(self, symbol: str) -> AsyncIterator[OrderBook]:
        """订阅订单簿"""
        if not self.exchange.has['ws']:
            raise NotImplementedError("WebSocket not supported")

        async for data in self.exchange.watch_order_book(symbol):
            yield self._convert_orderbook_data(data, symbol)

    async def _convert_orderbook_data(self, data: Dict, symbol: str) -> OrderBook:
        """转换订单簿数据"""
        bids = [Price(Decimal(str(p)), Decimal(str(v)), datetime.now())
                for p, v in data['bids']]
        asks = [Price(Decimal(str(p)), Decimal(str(v)), datetime.now())
                for p, v in data['asks']]

        return OrderBook(
            symbol=Symbol.from_string(symbol, self.metadata.name),
            bids=bids,
            asks=asks,
            timestamp=datetime.fromtimestamp(data['timestamp'] / 1000),
            exchange=self.metadata.name
        )
```

## 3. 自定义插件框架

### 3.1 自定义插件基类

```python
class CustomExchangePlugin(ExchangePluginInterface):
    """自定义交易所插件基类"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws_connections: Dict[str, Any] = {}
        self.auth_handler: Optional[AuthHandler] = None

    async def initialize(self) -> bool:
        """初始化自定义插件"""
        try:
            # 创建HTTP会话
            timeout = aiohttp.ClientTimeout(total=self.config.get('timeout', 30))
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self._get_default_headers()
            )

            # 初始化认证处理器
            if self.metadata.requires_credentials:
                self.auth_handler = self._create_auth_handler()

            # 设置速率限制器
            rate_limit = self.config.get('rate_limit', 1000)
            self.rate_limiter = RateLimiter(rate_limit)

            return True

        except Exception as e:
            logger.error(f"Custom plugin initialization failed: {e}")
            return False

    def _get_default_headers(self) -> Dict[str, str]:
        """获取默认请求头"""
        return {
            'User-Agent': 'FundingRateArb/1.0',
            'Content-Type': 'application/json',
        }

    def _create_auth_handler(self) -> 'AuthHandler':
        """创建认证处理器"""
        return AuthHandler(
            api_key=self.config.get('api_key', ''),
            secret_key=self.config.get('secret_key', ''),
            passphrase=self.config.get('passphrase', '')
        )

    async def _make_request(self,
                          method: str,
                          endpoint: str,
                          params: Dict = None,
                          data: Dict = None,
                          auth_required: bool = False) -> Dict:
        """发送HTTP请求"""
        await self.rate_limiter.acquire()

        url = self._build_url(endpoint)
        headers = self._get_default_headers()

        if auth_required and self.auth_handler:
            auth_headers = await self.auth_handler.get_auth_headers(
                method, endpoint, params, data
            )
            headers.update(auth_headers)

        try:
            async with self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                headers=headers
            ) as response:

                if response.status != 200:
                    error_text = await response.text()
                    raise ExchangeAPIError(f"HTTP {response.status}: {error_text}")

                return await response.json()

        except aiohttp.ClientError as e:
            raise ExchangeConnectionError(f"Connection error: {e}")

    def _build_url(self, endpoint: str) -> str:
        """构建请求URL"""
        base_url = self.config.get('base_url', '')
        return f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"

    # 子类需要实现的抽象方法
    @abstractmethod
    def _get_symbols_endpoint(self) -> str:
        """获取交易对列表的端点"""
        pass

    @abstractmethod
    def _get_orderbook_endpoint(self, symbol: str) -> str:
        """获取订单簿的端点"""
        pass

    @abstractmethod
    def _get_funding_rate_endpoint(self, symbol: str) -> str:
        """获取资金费率的端点"""
        pass

    @abstractmethod
    def _parse_symbol_data(self, data: Dict) -> List[Symbol]:
        """解析交易对数据"""
        pass

    @abstractmethod
    def _parse_orderbook_data(self, data: Dict, symbol: str) -> OrderBook:
        """解析订单簿数据"""
        pass

    @abstractmethod
    def _parse_funding_rate_data(self, data: Dict, symbol: str) -> FundingRate:
        """解析资金费率数据"""
        pass

class AuthHandler:
    """认证处理器"""

    def __init__(self, api_key: str, secret_key: str, passphrase: str = ''):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase

    async def get_auth_headers(self,
                             method: str,
                             endpoint: str,
                             params: Dict = None,
                             data: Dict = None) -> Dict[str, str]:
        """获取认证头"""
        timestamp = str(int(time.time() * 1000))

        # 构建签名字符串
        sign_string = self._build_sign_string(method, endpoint, params, data, timestamp)

        # 计算签名
        signature = self._calculate_signature(sign_string)

        return {
            'X-API-KEY': self.api_key,
            'X-TIMESTAMP': timestamp,
            'X-SIGNATURE': signature,
        }

    def _build_sign_string(self,
                          method: str,
                          endpoint: str,
                          params: Dict,
                          data: Dict,
                          timestamp: str) -> str:
        """构建签名字符串 - 需要子类根据具体交易所实现"""
        raise NotImplementedError("Subclass must implement _build_sign_string")

    def _calculate_signature(self, sign_string: str) -> str:
        """计算签名 - 需要子类根据具体交易所实现"""
        raise NotImplementedError("Subclass must implement _calculate_signature")
```

### 3.2 具体交易所实现示例

```python
class CustomBinancePlugin(CustomExchangePlugin):
    """自定义Binance插件示例"""

    def __init__(self, config: Dict[str, Any]):
        config['base_url'] = config.get('base_url', 'https://fapi.binance.com')
        super().__init__(config)

        self.metadata = ExchangeMetadata(
            name="binance_custom",
            display_name="Binance (Custom)",
            version="1.0.0",
            plugin_type=PluginType.CUSTOM,
            capabilities=ExchangeCapabilities(
                spot_trading=True,
                futures_trading=True,
                orderbook_stream=True,
                trades_stream=True,
                funding_rate=True,
            ),
            rate_limits={'default': 1200},
            supported_symbols=[],
            base_currencies=[],
            quote_currencies=[],
            min_trade_amounts={},
            fee_structure={'taker': 0.0004, 'maker': 0.0002},
        )

    def _get_symbols_endpoint(self) -> str:
        return "fapi/v1/exchangeInfo"

    def _get_orderbook_endpoint(self, symbol: str) -> str:
        return f"fapi/v1/depth?symbol={symbol}&limit=100"

    def _get_funding_rate_endpoint(self, symbol: str) -> str:
        return f"fapi/v1/fundingRate?symbol={symbol}"

    def _parse_symbol_data(self, data: Dict) -> List[Symbol]:
        """解析Binance交易对数据"""
        symbols = []

        for symbol_info in data.get('symbols', []):
            if symbol_info['status'] == 'TRADING':
                symbols.append(Symbol(
                    base=symbol_info['baseAsset'],
                    quote=symbol_info['quoteAsset'],
                    exchange=self.metadata.name,
                    symbol=symbol_info['symbol'],
                    contract_type='futures'
                ))

        return symbols

    def _parse_orderbook_data(self, data: Dict, symbol: str) -> OrderBook:
        """解析Binance订单簿数据"""
        bids = [Price(Decimal(p), Decimal(v), datetime.now())
                for p, v in data['bids']]
        asks = [Price(Decimal(p), Decimal(v), datetime.now())
                for p, v in data['asks']]

        return OrderBook(
            symbol=Symbol.from_string(symbol, self.metadata.name),
            bids=bids,
            asks=asks,
            timestamp=datetime.now(),
            exchange=self.metadata.name
        )

    def _parse_funding_rate_data(self, data: Dict, symbol: str) -> FundingRate:
        """解析Binance资金费率数据"""
        return FundingRate(
            exchange=self.metadata.name,
            symbol=symbol,
            rate=Decimal(str(data['fundingRate'])),
            timestamp=datetime.fromtimestamp(data['fundingTime'] / 1000),
            next_funding_time=datetime.fromtimestamp(data['fundingTime'] / 1000 + 8*3600),
            funding_interval=8 * 3600
        )

    # 实现认证处理器
    class BinanceAuthHandler(AuthHandler):
        """Binance认证处理器"""

        def _build_sign_string(self, method: str, endpoint: str,
                              params: Dict, data: Dict, timestamp: str) -> str:
            query_string = ""

            if params:
                query_string = "&".join([f"{k}={v}" for k, v in params.items()])

            if data:
                body_string = json.dumps(data, separators=(',', ':'))
                query_string = f"{query_string}&{body_string}" if query_string else body_string

            query_string = f"{query_string}&timestamp={timestamp}" if query_string else f"timestamp={timestamp}"

            return query_string

        def _calculate_signature(self, sign_string: str) -> str:
            import hmac
            import hashlib

            return hmac.new(
                self.secret_key.encode('utf-8'),
                sign_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()

    def _create_auth_handler(self) -> AuthHandler:
        return self.BinanceAuthHandler(
            api_key=self.config.get('api_key', ''),
            secret_key=self.config.get('secret_key', ''),
            passphrase=self.config.get('passphrase', '')
        )
```

## 4. 插件管理系统

### 4.1 插件注册与发现

```python
class PluginRegistry:
    """插件注册中心"""

    def __init__(self):
        self.plugins: Dict[str, Type[ExchangePluginInterface]] = {}
        self.metadata_cache: Dict[str, ExchangeMetadata] = {}

    def register_plugin(self,
                       name: str,
                       plugin_class: Type[ExchangePluginInterface],
                       metadata: ExchangeMetadata = None):
        """注册插件"""
        self.plugins[name] = plugin_class

        if metadata:
            self.metadata_cache[name] = metadata

        logger.info(f"Plugin registered: {name}")

    def unregister_plugin(self, name: str):
        """注销插件"""
        self.plugins.pop(name, None)
        self.metadata_cache.pop(name, None)

        logger.info(f"Plugin unregistered: {name}")

    def get_plugin_class(self, name: str) -> Optional[Type[ExchangePluginInterface]]:
        """获取插件类"""
        return self.plugins.get(name)

    def get_available_plugins(self) -> List[str]:
        """获取可用插件列表"""
        return list(self.plugins.keys())

    def get_plugin_metadata(self, name: str) -> Optional[ExchangeMetadata]:
        """获取插件元数据"""
        return self.metadata_cache.get(name)

    def discover_plugins(self, plugin_dir: str):
        """自动发现插件"""
        import importlib.util
        import os

        for filename in os.listdir(plugin_dir):
            if filename.endswith('_plugin.py'):
                plugin_name = filename[:-10]  # 移除 '_plugin.py'
                file_path = os.path.join(plugin_dir, filename)

                try:
                    # 动态导入插件模块
                    spec = importlib.util.spec_from_file_location(plugin_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # 查找插件类
                    plugin_class = getattr(module, 'Plugin', None)
                    if plugin_class and issubclass(plugin_class, ExchangePluginInterface):
                        self.register_plugin(plugin_name, plugin_class)

                except Exception as e:
                    logger.error(f"Failed to discover plugin {plugin_name}: {e}")

class PluginLoader:
    """插件加载器"""

    def __init__(self, registry: PluginRegistry):
        self.registry = registry
        self.loaded_plugins: Dict[str, ExchangePluginInterface] = {}

    async def load_plugin(self, name: str, config: Dict[str, Any]) -> ExchangePluginInterface:
        """加载插件"""
        plugin_class = self.registry.get_plugin_class(name)
        if not plugin_class:
            raise ValueError(f"Plugin not found: {name}")

        try:
            # 创建插件实例
            plugin = plugin_class(config)

            # 初始化插件
            if await plugin.initialize():
                self.loaded_plugins[name] = plugin
                logger.info(f"Plugin loaded: {name}")
                return plugin
            else:
                raise RuntimeError(f"Plugin initialization failed: {name}")

        except Exception as e:
            logger.error(f"Failed to load plugin {name}: {e}")
            raise

    async def unload_plugin(self, name: str):
        """卸载插件"""
        plugin = self.loaded_plugins.get(name)
        if plugin:
            await plugin.disconnect()
            del self.loaded_plugins[name]
            logger.info(f"Plugin unloaded: {name}")

    def get_loaded_plugin(self, name: str) -> Optional[ExchangePluginInterface]:
        """获取已加载的插件"""
        return self.loaded_plugins.get(name)

    def get_loaded_plugins(self) -> Dict[str, ExchangePluginInterface]:
        """获取所有已加载的插件"""
        return self.loaded_plugins.copy()

class PluginManager:
    """插件管理器"""

    def __init__(self):
        self.registry = PluginRegistry()
        self.loader = PluginLoader(self.registry)
        self.config_manager: Optional[ConfigManager] = None

    def set_config_manager(self, config_manager: ConfigManager):
        """设置配置管理器"""
        self.config_manager = config_manager

    async def initialize(self):
        """初始化插件管理器"""
        # 注册内置插件
        self._register_builtin_plugins()

        # 发现外部插件
        plugin_dir = "plugins"
        if os.path.exists(plugin_dir):
            self.registry.discover_plugins(plugin_dir)

        # 加载配置中的插件
        if self.config_manager:
            await self._load_configured_plugins()

    def _register_builtin_plugins(self):
        """注册内置插件"""
        # 注册CCXT支持的交易所
        ccxt_exchanges = [
            'binance', 'okx', 'bybit', 'huobi', 'bitget', 'kucoin',
            'gate', 'mexc', 'cryptocom', 'bitfinex'
        ]

        for exchange_id in ccxt_exchanges:
            if hasattr(ccxt, exchange_id):
                self.registry.register_plugin(
                    f"{exchange_id}_ccxt",
                    CCXTAdapter,
                    self._create_ccxt_metadata(exchange_id)
                )

        # 注册自定义插件
        self.registry.register_plugin("binance_custom", CustomBinancePlugin)

    def _create_ccxt_metadata(self, exchange_id: str) -> ExchangeMetadata:
        """创建CCXT插件元数据"""
        return ExchangeMetadata(
            name=f"{exchange_id}_ccxt",
            display_name=exchange_id.capitalize(),
            version="1.0.0",
            plugin_type=PluginType.CCXT,
            capabilities=ExchangeCapabilities(),  # 实际值需要从CCXT获取
            rate_limits={'default': 1000},
            supported_symbols=[],
            base_currencies=[],
            quote_currencies=[],
            min_trade_amounts={},
            fee_structure={},
        )

    async def _load_configured_plugins(self):
        """加载配置中的插件"""
        if not self.config_manager:
            return

        config = self.config_manager.load_config()

        for exchange_config in config.exchanges:
            try:
                plugin_name = f"{exchange_config.name}_ccxt"
                if exchange_config.name in self.registry.get_available_plugins():
                    plugin_name = exchange_config.name

                await self.loader.load_plugin(plugin_name, {
                    'exchange_id': exchange_config.name,
                    'api_key': exchange_config.api_key,
                    'secret_key': exchange_config.secret_key,
                    'passphrase': exchange_config.passphrase,
                    'sandbox': exchange_config.sandbox,
                    'timeout': exchange_config.timeout,
                    'rate_limit': exchange_config.rate_limit,
                })

            except Exception as e:
                logger.error(f"Failed to load exchange plugin {exchange_config.name}: {e}")

    async def reload_plugin(self, name: str):
        """重新加载插件"""
        # 卸载现有插件
        await self.loader.unload_plugin(name)

        # 重新加载插件
        if self.config_manager:
            config = self.config_manager.get_exchange_config(name)
            if config:
                await self.loader.load_plugin(name, config.__dict__)

    def get_plugin(self, name: str) -> Optional[ExchangePluginInterface]:
        """获取插件实例"""
        return self.loader.get_loaded_plugin(name)

    def get_all_plugins(self) -> Dict[str, ExchangePluginInterface]:
        """获取所有插件实例"""
        return self.loader.get_loaded_plugins()

    async def health_check_all(self) -> Dict[str, Dict]:
        """检查所有插件健康状态"""
        results = {}

        for name, plugin in self.loader.get_loaded_plugins().items():
            try:
                results[name] = await plugin.health_check()
            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'error': str(e),
                    'last_check': datetime.now().isoformat()
                }

        return results

    async def shutdown(self):
        """关闭插件管理器"""
        tasks = []

        for name in list(self.loader.loaded_plugins.keys()):
            tasks.append(self.loader.unload_plugin(name))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        logger.info("Plugin manager shutdown complete")
```

## 5. 错误处理与重连机制

### 5.1 异常定义

```python
class ExchangeError(Exception):
    """交易所异常基类"""
    pass

class ExchangeAPIError(ExchangeError):
    """API错误"""
    def __init__(self, message: str, status_code: int = None, response: Dict = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response

class ExchangeConnectionError(ExchangeError):
    """连接错误"""
    pass

class ExchangeAuthError(ExchangeError):
    """认证错误"""
    pass

class ExchangeRateLimitError(ExchangeError):
    """速率限制错误"""
    def __init__(self, message: str, retry_after: int = None):
        super().__init__(message)
        self.retry_after = retry_after
```

### 5.2 重连机制

```python
class ReconnectManager:
    """重连管理器"""

    def __init__(self,
                 plugin: ExchangePluginInterface,
                 max_retries: int = 5,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0):
        self.plugin = plugin
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retry_count = 0
        self.is_reconnecting = False

    async def handle_disconnect(self):
        """处理断线"""
        if self.is_reconnecting:
            return

        self.is_reconnecting = True
        self.plugin.connection_status = ConnectionStatus.RECONNECTING

        try:
            while self.retry_count < self.max_retries:
                delay = min(self.base_delay * (2 ** self.retry_count), self.max_delay)

                logger.info(f"Attempting reconnection in {delay}s (attempt {self.retry_count + 1}/{self.max_retries})")
                await asyncio.sleep(delay)

                if await self.plugin.connect():
                    logger.info("Reconnection successful")
                    self.retry_count = 0
                    self.is_reconnecting = False
                    return True

                self.retry_count += 1

            logger.error(f"Reconnection failed after {self.max_retries} attempts")
            self.plugin.connection_status = ConnectionStatus.ERROR
            return False

        except Exception as e:
            logger.error(f"Reconnection error: {e}")
            self.plugin.connection_status = ConnectionStatus.ERROR
            return False
        finally:
            self.is_reconnecting = False

class RateLimiter:
    """速率限制器"""

    def __init__(self, rate: float):
        self.rate = rate  # requests per second
        self.tokens = rate
        self.last_update = time.time()
        self.lock = asyncio.Lock()

    async def acquire(self):
        """获取令牌"""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return

            # 计算等待时间
            wait_time = (1 - self.tokens) / self.rate
            await asyncio.sleep(wait_time)
            self.tokens = 0
```

这个交易所插件化架构设计提供了完整的可扩展框架，支持CCXT集成、自定义实现、热插拔等功能，确保系统能够灵活适应各种交易所的接入需求。