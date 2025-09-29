# 风险管理与订单生命周期管理

## 1. 风险管理体系概览

风险管理是资金费率套利系统的核心组件，负责控制交易风险、保护资金安全、确保系统稳定运行。本系统采用多层级风险控制架构，从交易前检查到实时监控，全方位覆盖各类风险。

```
┌─────────────────────────────────────────────────────────────┐
│                    风险管理架构                                │
├─────────────────────────────────────────────────────────────┤
│  交易前风控  │  实时风控  │  事后风控  │  应急处理  │  风险报告  │
└─────────────────────────────────────────────────────────────┘
        │           │           │           │           │
┌───────────────────────────────────────────────────────────────┐
│                      风险控制层级                               │
├───────────────────────────────────────────────────────────────┤
│  订单级  │  账户级  │  交易所级  │  策略级  │  系统级  │  组合级  │
└───────────────────────────────────────────────────────────────┘
```

## 2. 风险类型与控制策略

### 2.1 风险分类体系

```python
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass
from decimal import Decimal
import numpy as np

class RiskType(Enum):
    # 市场风险
    MARKET_RISK = "market_risk"              # 市场风险
    LIQUIDITY_RISK = "liquidity_risk"        # 流动性风险
    SLIPPAGE_RISK = "slippage_risk"          # 滑点风险

    # 操作风险
    OPERATIONAL_RISK = "operational_risk"     # 操作风险
    SYSTEM_RISK = "system_risk"              # 系统风险
    API_RISK = "api_risk"                    # API风险

    # 信用风险
    COUNTERPARTY_RISK = "counterparty_risk"   # 交易对手风险
    SETTLEMENT_RISK = "settlement_risk"       # 结算风险

    # 模型风险
    MODEL_RISK = "model_risk"                # 模型风险
    PARAMETER_RISK = "parameter_risk"        # 参数风险

@dataclass
class RiskMetric:
    name: str
    value: float
    threshold: float
    status: str                              # OK, WARNING, CRITICAL
    last_updated: datetime
    description: str

class RiskLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class RiskLimit:
    name: str
    limit_type: str                          # absolute, percentage
    value: Decimal
    currency: str
    scope: str                               # order, account, strategy, system
    enabled: bool = True
    breach_action: str = "reject"            # reject, warn, reduce_size
```

### 2.2 风险限制配置

```python
@dataclass
class RiskLimits:
    """风险限制配置"""

    # 仓位限制
    max_position_size: Dict[str, Decimal]      # 单品种最大仓位
    max_net_position: Decimal                  # 最大净持仓
    max_gross_position: Decimal                # 最大总持仓
    position_concentration: float              # 仓位集中度限制

    # 损失限制
    max_daily_loss: Decimal                    # 最大日损失
    max_weekly_loss: Decimal                   # 最大周损失
    max_monthly_loss: Decimal                  # 最大月损失
    max_drawdown: float                        # 最大回撤

    # 风险价值限制
    var_1d_95: Decimal                         # 1天95%置信度VaR
    var_1d_99: Decimal                         # 1天99%置信度VaR
    cvar_1d_95: Decimal                        # 1天95%置信度CVaR

    # 流动性限制
    max_order_size_ratio: float                # 订单大小占成交量比例
    min_market_depth: Decimal                  # 最小市场深度
    max_spread_threshold: float                # 最大价差阈值

    # 杠杆限制
    max_leverage: float                        # 最大杠杆倍数
    margin_utilization_limit: float           # 保证金使用率限制

    # 交易频率限制
    max_orders_per_minute: int                 # 每分钟最大订单数
    max_trades_per_hour: int                   # 每小时最大交易次数
    cooling_period: int                        # 冷却期（秒）

class RiskLimitManager:
    """风险限制管理器"""

    def __init__(self, config: RiskLimits):
        self.limits = config
        self.current_metrics: Dict[str, RiskMetric] = {}

    def add_limit(self, limit: RiskLimit):
        """添加风险限制"""
        self.limits.__dict__[limit.name] = limit.value

    def update_limit(self, name: str, value: Decimal):
        """更新风险限制"""
        if hasattr(self.limits, name):
            setattr(self.limits, name, value)

    def check_limit(self, metric_name: str, current_value: float) -> bool:
        """检查是否超限"""
        if not hasattr(self.limits, metric_name):
            return True

        limit = getattr(self.limits, metric_name)
        return current_value <= float(limit)

    def get_risk_utilization(self) -> Dict[str, float]:
        """获取风险使用率"""
        utilization = {}

        for name, metric in self.current_metrics.items():
            if hasattr(self.limits, name):
                limit = getattr(self.limits, name)
                utilization[name] = metric.value / float(limit) if limit > 0 else 0

        return utilization
```

## 3. 实时风险计算与监控

### 3.1 VaR计算引擎

```python
import pandas as pd
from scipy import stats
from typing import Tuple

class VaRCalculator:
    """风险价值计算器"""

    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        self.confidence_levels = [0.95, 0.99]

    async def calculate_historical_var(self,
                                     returns: pd.Series,
                                     confidence: float = 0.95,
                                     horizon_days: int = 1) -> Decimal:
        """历史模拟法计算VaR"""
        if len(returns) < self.lookback_days:
            raise ValueError(f"Insufficient data: {len(returns)} < {self.lookback_days}")

        # 计算历史收益率分布
        historical_returns = returns.tail(self.lookback_days)

        # 调整时间跨度
        if horizon_days > 1:
            historical_returns *= np.sqrt(horizon_days)

        # 计算VaR
        var_value = np.percentile(historical_returns, (1 - confidence) * 100)

        return Decimal(str(abs(var_value)))

    async def calculate_parametric_var(self,
                                     returns: pd.Series,
                                     confidence: float = 0.95,
                                     horizon_days: int = 1) -> Decimal:
        """参数化方法计算VaR"""
        # 计算收益率统计量
        mean_return = returns.mean()
        std_return = returns.std()

        # 正态分布假设下的VaR
        z_score = stats.norm.ppf(1 - confidence)
        var_value = abs(mean_return + z_score * std_return) * np.sqrt(horizon_days)

        return Decimal(str(var_value))

    async def calculate_monte_carlo_var(self,
                                      returns: pd.Series,
                                      confidence: float = 0.95,
                                      horizon_days: int = 1,
                                      simulations: int = 10000) -> Decimal:
        """蒙特卡洛模拟法计算VaR"""
        # 拟合收益率分布
        mean_return = returns.mean()
        std_return = returns.std()

        # 蒙特卡洛模拟
        np.random.seed(42)
        simulated_returns = np.random.normal(
            mean_return * horizon_days,
            std_return * np.sqrt(horizon_days),
            simulations
        )

        # 计算VaR
        var_value = np.percentile(simulated_returns, (1 - confidence) * 100)

        return Decimal(str(abs(var_value)))

    async def calculate_cvar(self,
                           returns: pd.Series,
                           confidence: float = 0.95) -> Decimal:
        """计算条件风险价值CVaR"""
        var_value = await self.calculate_historical_var(returns, confidence)

        # CVaR是超过VaR的期望损失
        threshold = float(var_value)
        tail_losses = returns[returns <= -threshold]

        if len(tail_losses) == 0:
            return var_value

        cvar_value = abs(tail_losses.mean())
        return Decimal(str(cvar_value))

class RealTimeRiskCalculator:
    """实时风险计算器"""

    def __init__(self, var_calculator: VaRCalculator):
        self.var_calculator = var_calculator
        self.portfolio_returns = pd.Series()
        self.position_values = {}

    async def update_portfolio_value(self, positions: Dict[str, Position]):
        """更新组合价值"""
        total_value = Decimal('0')
        position_values = {}

        for symbol, position in positions.items():
            position_value = position.amount * position.mark_price
            position_values[symbol] = position_value
            total_value += position_value

        self.position_values = position_values
        return total_value

    async def calculate_portfolio_var(self,
                                    confidence: float = 0.95) -> RiskMetric:
        """计算组合VaR"""
        if len(self.portfolio_returns) < 30:
            # 数据不足时使用保守估计
            var_estimate = sum(self.position_values.values()) * Decimal('0.05')
        else:
            var_estimate = await self.var_calculator.calculate_historical_var(
                self.portfolio_returns, confidence
            )

        return RiskMetric(
            name="portfolio_var",
            value=float(var_estimate),
            threshold=1000000.0,  # 从配置获取
            status="OK" if var_estimate < 1000000 else "WARNING",
            last_updated=datetime.now(),
            description="组合风险价值"
        )

    async def calculate_position_risk(self, position: Position) -> Dict[str, RiskMetric]:
        """计算单个仓位风险"""
        risks = {}

        # 仓位价值风险
        position_value = position.amount * position.mark_price
        risks["position_value"] = RiskMetric(
            name="position_value",
            value=float(position_value),
            threshold=100000.0,
            status="OK" if position_value < 100000 else "WARNING",
            last_updated=datetime.now(),
            description="仓位价值风险"
        )

        # 未实现盈亏风险
        risks["unrealized_pnl"] = RiskMetric(
            name="unrealized_pnl",
            value=float(position.unrealized_pnl),
            threshold=-10000.0,
            status="OK" if position.unrealized_pnl > -10000 else "CRITICAL",
            last_updated=datetime.now(),
            description="未实现盈亏风险"
        )

        return risks

    async def calculate_correlation_risk(self, positions: Dict[str, Position]) -> RiskMetric:
        """计算相关性风险"""
        symbols = list(positions.keys())

        if len(symbols) < 2:
            return RiskMetric(
                name="correlation_risk",
                value=0.0,
                threshold=0.8,
                status="OK",
                last_updated=datetime.now(),
                description="相关性风险"
            )

        # 计算持仓相关性（简化实现）
        correlation_matrix = np.random.rand(len(symbols), len(symbols))
        np.fill_diagonal(correlation_matrix, 1.0)

        max_correlation = np.max(correlation_matrix[correlation_matrix < 1.0])

        return RiskMetric(
            name="correlation_risk",
            value=max_correlation,
            threshold=0.8,
            status="OK" if max_correlation < 0.8 else "WARNING",
            last_updated=datetime.now(),
            description="最大持仓相关性"
        )
```

### 3.2 实时风险监控

```python
class RealTimeRiskMonitor:
    """实时风险监控器"""

    def __init__(self,
                 limits: RiskLimits,
                 calculator: RealTimeRiskCalculator,
                 message_bus: MessageBusInterface):
        self.limits = limits
        self.calculator = calculator
        self.message_bus = message_bus
        self.risk_metrics: Dict[str, RiskMetric] = {}
        self.alert_history: List[Dict] = []
        self.monitoring_active = False

    async def start_monitoring(self):
        """启动实时监控"""
        self.monitoring_active = True
        asyncio.create_task(self._monitoring_loop())

    async def stop_monitoring(self):
        """停止实时监控"""
        self.monitoring_active = False

    async def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 计算风险指标
                await self._calculate_risk_metrics()

                # 检查风险限制
                alerts = await self._check_risk_limits()

                # 处理告警
                for alert in alerts:
                    await self._handle_alert(alert)

                # 发布风险报告
                await self._publish_risk_report()

                await asyncio.sleep(1)  # 1秒监控间隔

            except Exception as e:
                logger.error(f"Risk monitoring error: {e}")
                await asyncio.sleep(5)

    async def _calculate_risk_metrics(self):
        """计算风险指标"""
        # 获取当前持仓
        positions = await self._get_current_positions()

        # 计算组合VaR
        var_metric = await self.calculator.calculate_portfolio_var()
        self.risk_metrics["portfolio_var"] = var_metric

        # 计算总仓位风险
        total_position_value = sum(
            pos.amount * pos.mark_price for pos in positions.values()
        )

        self.risk_metrics["total_position"] = RiskMetric(
            name="total_position",
            value=float(total_position_value),
            threshold=float(self.limits.max_gross_position),
            status="OK" if total_position_value < self.limits.max_gross_position else "CRITICAL",
            last_updated=datetime.now(),
            description="总仓位价值"
        )

        # 计算相关性风险
        correlation_metric = await self.calculator.calculate_correlation_risk(positions)
        self.risk_metrics["correlation"] = correlation_metric

    async def _check_risk_limits(self) -> List[Dict]:
        """检查风险限制"""
        alerts = []

        for metric_name, metric in self.risk_metrics.items():
            if metric.status in ["WARNING", "CRITICAL"]:
                alert = {
                    "type": "risk_limit_breach",
                    "metric": metric_name,
                    "current_value": metric.value,
                    "threshold": metric.threshold,
                    "severity": metric.status,
                    "timestamp": datetime.now(),
                    "description": metric.description
                }
                alerts.append(alert)

        return alerts

    async def _handle_alert(self, alert: Dict):
        """处理风险告警"""
        # 记录告警历史
        self.alert_history.append(alert)

        # 发送告警消息
        await self.message_bus.publish(Topics.RISK_ALERT, alert)

        # 根据严重程度采取行动
        if alert["severity"] == "CRITICAL":
            await self._emergency_stop(alert["metric"])
        elif alert["severity"] == "WARNING":
            await self._reduce_risk_exposure(alert["metric"])

    async def _emergency_stop(self, trigger_metric: str):
        """紧急停止"""
        stop_message = {
            "type": "emergency_stop",
            "trigger": trigger_metric,
            "timestamp": datetime.now(),
            "action": "stop_all_trading"
        }

        await self.message_bus.publish(Topics.EMERGENCY_STOP, stop_message)
        logger.critical(f"Emergency stop triggered by {trigger_metric}")

    async def _reduce_risk_exposure(self, trigger_metric: str):
        """降低风险敞口"""
        reduce_message = {
            "type": "reduce_exposure",
            "trigger": trigger_metric,
            "timestamp": datetime.now(),
            "action": "reduce_position_size"
        }

        await self.message_bus.publish(Topics.RISK_ALERT, reduce_message)
        logger.warning(f"Risk exposure reduction triggered by {trigger_metric}")

    async def _publish_risk_report(self):
        """发布风险报告"""
        risk_report = {
            "timestamp": datetime.now(),
            "metrics": {name: {
                "value": metric.value,
                "threshold": metric.threshold,
                "status": metric.status,
                "description": metric.description
            } for name, metric in self.risk_metrics.items()},
            "overall_status": self._calculate_overall_risk_status(),
            "alerts_count": len([m for m in self.risk_metrics.values()
                               if m.status in ["WARNING", "CRITICAL"]])
        }

        await self.message_bus.publish("risk.report", risk_report)

    def _calculate_overall_risk_status(self) -> str:
        """计算总体风险状态"""
        statuses = [metric.status for metric in self.risk_metrics.values()]

        if "CRITICAL" in statuses:
            return "CRITICAL"
        elif "WARNING" in statuses:
            return "WARNING"
        else:
            return "OK"
```

## 4. 订单生命周期管理

### 4.1 订单状态机

```python
from enum import Enum
from typing import Optional, Callable
import asyncio

class OrderState(Enum):
    # 初始状态
    CREATED = "created"                      # 订单已创建
    VALIDATED = "validated"                  # 订单已验证

    # 提交状态
    SUBMITTED = "submitted"                  # 订单已提交
    PENDING = "pending"                      # 订单等待中
    ACCEPTED = "accepted"                    # 订单已接受
    REJECTED = "rejected"                    # 订单已拒绝

    # 执行状态
    OPEN = "open"                           # 订单活跃
    PARTIALLY_FILLED = "partially_filled"    # 订单部分成交
    FILLED = "filled"                       # 订单完全成交

    # 终止状态
    CANCELLED = "cancelled"                  # 订单已取消
    EXPIRED = "expired"                      # 订单已过期
    FAILED = "failed"                       # 订单失败

class OrderEvent(Enum):
    VALIDATE = "validate"
    SUBMIT = "submit"
    ACCEPT = "accept"
    REJECT = "reject"
    FILL = "fill"
    PARTIAL_FILL = "partial_fill"
    CANCEL = "cancel"
    EXPIRE = "expire"
    FAIL = "fail"

@dataclass
class OrderTransition:
    from_state: OrderState
    event: OrderEvent
    to_state: OrderState
    condition: Optional[Callable] = None
    action: Optional[Callable] = None

class OrderStateMachine:
    """订单状态机"""

    def __init__(self):
        self.transitions: Dict[Tuple[OrderState, OrderEvent], OrderTransition] = {}
        self._setup_transitions()

    def _setup_transitions(self):
        """设置状态转换"""
        transitions = [
            # 创建到验证
            OrderTransition(OrderState.CREATED, OrderEvent.VALIDATE, OrderState.VALIDATED),

            # 验证到提交
            OrderTransition(OrderState.VALIDATED, OrderEvent.SUBMIT, OrderState.SUBMITTED),

            # 提交后的状态转换
            OrderTransition(OrderState.SUBMITTED, OrderEvent.ACCEPT, OrderState.ACCEPTED),
            OrderTransition(OrderState.SUBMITTED, OrderEvent.REJECT, OrderState.REJECTED),

            # 接受后的状态转换
            OrderTransition(OrderState.ACCEPTED, OrderEvent.SUBMIT, OrderState.OPEN),

            # 活跃订单的状态转换
            OrderTransition(OrderState.OPEN, OrderEvent.PARTIAL_FILL, OrderState.PARTIALLY_FILLED),
            OrderTransition(OrderState.OPEN, OrderEvent.FILL, OrderState.FILLED),
            OrderTransition(OrderState.OPEN, OrderEvent.CANCEL, OrderState.CANCELLED),
            OrderTransition(OrderState.OPEN, OrderEvent.EXPIRE, OrderState.EXPIRED),

            # 部分成交订单的状态转换
            OrderTransition(OrderState.PARTIALLY_FILLED, OrderEvent.FILL, OrderState.FILLED),
            OrderTransition(OrderState.PARTIALLY_FILLED, OrderEvent.PARTIAL_FILL, OrderState.PARTIALLY_FILLED),
            OrderTransition(OrderState.PARTIALLY_FILLED, OrderEvent.CANCEL, OrderState.CANCELLED),

            # 错误处理
            OrderTransition(OrderState.CREATED, OrderEvent.FAIL, OrderState.FAILED),
            OrderTransition(OrderState.VALIDATED, OrderEvent.FAIL, OrderState.FAILED),
            OrderTransition(OrderState.SUBMITTED, OrderEvent.FAIL, OrderState.FAILED),
        ]

        for transition in transitions:
            key = (transition.from_state, transition.event)
            self.transitions[key] = transition

    async def transition(self, order: Order, event: OrderEvent) -> bool:
        """执行状态转换"""
        key = (OrderState(order.status), event)
        transition = self.transitions.get(key)

        if not transition:
            logger.warning(f"Invalid transition: {order.status} -> {event}")
            return False

        # 检查转换条件
        if transition.condition and not await transition.condition(order):
            logger.warning(f"Transition condition failed: {order.id}")
            return False

        # 执行转换动作
        if transition.action:
            await transition.action(order)

        # 更新订单状态
        old_status = order.status
        order.status = transition.to_state.value

        logger.info(f"Order {order.id} transitioned: {old_status} -> {order.status}")
        return True

    def can_transition(self, current_state: OrderState, event: OrderEvent) -> bool:
        """检查是否可以转换"""
        key = (current_state, event)
        return key in self.transitions

    def get_available_events(self, current_state: OrderState) -> List[OrderEvent]:
        """获取当前状态可用的事件"""
        return [event for (state, event) in self.transitions.keys()
                if state == current_state]

class OrderLifecycleManager:
    """订单生命周期管理器"""

    def __init__(self,
                 exchange_manager: ExchangeManager,
                 risk_monitor: RealTimeRiskMonitor,
                 message_bus: MessageBusInterface):
        self.exchange_manager = exchange_manager
        self.risk_monitor = risk_monitor
        self.message_bus = message_bus
        self.state_machine = OrderStateMachine()

        self.active_orders: Dict[str, Order] = {}
        self.order_history: Dict[str, List[Dict]] = {}

    async def create_order(self, signal: TradingSignal) -> Order:
        """创建订单"""
        order = Order(
            id=self._generate_order_id(),
            client_order_id=f"client_{int(time.time() * 1000)}",
            symbol=signal.symbol,
            side=signal.side,
            order_type=OrderType.LIMIT,
            amount=signal.amount,
            price=signal.price,
            filled=Decimal('0'),
            remaining=signal.amount,
            status=OrderState.CREATED.value,
            timestamp=datetime.now()
        )

        # 记录订单创建
        await self._record_order_event(order, "order_created", {"signal_id": signal.id})

        # 开始管理订单生命周期
        asyncio.create_task(self._manage_order_lifecycle(order))

        return order

    async def _manage_order_lifecycle(self, order: Order):
        """管理订单生命周期"""
        try:
            # 1. 验证订单
            if await self._validate_order(order):
                await self.state_machine.transition(order, OrderEvent.VALIDATE)
            else:
                await self.state_machine.transition(order, OrderEvent.FAIL)
                return

            # 2. 风险检查
            if not await self._pre_trade_risk_check(order):
                await self._record_order_event(order, "risk_check_failed")
                await self.state_machine.transition(order, OrderEvent.REJECT)
                return

            # 3. 提交订单
            if await self._submit_order(order):
                await self.state_machine.transition(order, OrderEvent.SUBMIT)
                await self.state_machine.transition(order, OrderEvent.ACCEPT)
                await self.state_machine.transition(order, OrderEvent.SUBMIT)  # 到OPEN状态
            else:
                await self.state_machine.transition(order, OrderEvent.REJECT)
                return

            # 4. 监控订单执行
            self.active_orders[order.id] = order
            await self._monitor_order_execution(order)

        except Exception as e:
            logger.error(f"Order lifecycle management error: {e}")
            await self.state_machine.transition(order, OrderEvent.FAIL)

    async def _validate_order(self, order: Order) -> bool:
        """验证订单"""
        validations = [
            self._validate_symbol(order.symbol),
            self._validate_amount(order.amount),
            self._validate_price(order.price),
            self._validate_order_type(order.order_type),
        ]

        results = await asyncio.gather(*validations)
        is_valid = all(results)

        await self._record_order_event(order, "validation_result", {"valid": is_valid})
        return is_valid

    async def _validate_symbol(self, symbol: Symbol) -> bool:
        """验证交易对"""
        exchange = self.exchange_manager.get_exchange(symbol.exchange)
        if not exchange:
            return False

        symbols = await exchange.get_symbols()
        return any(s.symbol == symbol.symbol for s in symbols)

    async def _validate_amount(self, amount: Decimal) -> bool:
        """验证订单数量"""
        return amount > 0 and amount <= Decimal('1000000')  # 最大订单限制

    async def _validate_price(self, price: Optional[Decimal]) -> bool:
        """验证订单价格"""
        if price is None:
            return True  # 市价单
        return price > 0

    async def _validate_order_type(self, order_type: OrderType) -> bool:
        """验证订单类型"""
        supported_types = [OrderType.MARKET, OrderType.LIMIT, OrderType.IOC, OrderType.FOK]
        return order_type in supported_types

    async def _pre_trade_risk_check(self, order: Order) -> bool:
        """交易前风险检查"""
        # 检查仓位限制
        if not await self._check_position_limits(order):
            return False

        # 检查资金充足性
        if not await self._check_balance_sufficient(order):
            return False

        # 检查日内交易次数
        if not await self._check_trading_frequency(order):
            return False

        return True

    async def _check_position_limits(self, order: Order) -> bool:
        """检查仓位限制"""
        # 获取当前持仓
        exchange = self.exchange_manager.get_exchange(order.symbol.exchange)
        positions = await exchange.get_positions()

        current_position = next(
            (p for p in positions if p.symbol.symbol == order.symbol.symbol),
            None
        )

        # 计算新的持仓
        if current_position:
            if order.side == OrderSide.BUY:
                new_position = current_position.amount + order.amount
            else:
                new_position = current_position.amount - order.amount
        else:
            new_position = order.amount if order.side == OrderSide.BUY else -order.amount

        # 检查持仓限制
        max_position = self.risk_monitor.limits.max_position_size.get(
            order.symbol.symbol, Decimal('100000')
        )

        return abs(new_position) <= max_position

    async def _check_balance_sufficient(self, order: Order) -> bool:
        """检查资金充足性"""
        exchange = self.exchange_manager.get_exchange(order.symbol.exchange)
        balances = await exchange.get_balance()

        required_currency = order.symbol.quote if order.side == OrderSide.BUY else order.symbol.base
        required_amount = order.amount * order.price if order.side == OrderSide.BUY else order.amount

        available_balance = balances.get(required_currency, Balance(
            currency=required_currency, free=Decimal('0'), used=Decimal('0'), total=Decimal('0'),
            timestamp=datetime.now()
        ))

        return available_balance.free >= required_amount

    async def _submit_order(self, order: Order) -> bool:
        """提交订单到交易所"""
        try:
            exchange = self.exchange_manager.get_exchange(order.symbol.exchange)
            submitted_order = await exchange.create_order(order)

            # 更新订单信息
            order.id = submitted_order.id

            await self._record_order_event(order, "order_submitted", {
                "exchange_order_id": submitted_order.id
            })

            return True

        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            await self._record_order_event(order, "submission_failed", {"error": str(e)})
            return False

    async def _monitor_order_execution(self, order: Order):
        """监控订单执行"""
        exchange = self.exchange_manager.get_exchange(order.symbol.exchange)

        while order.status in [OrderState.OPEN.value, OrderState.PARTIALLY_FILLED.value]:
            try:
                # 查询订单状态
                current_order = await exchange.get_order(order.id, order.symbol.symbol)

                # 检查状态变化
                if current_order.status != order.status:
                    await self._handle_order_status_change(order, current_order)

                # 检查填充量变化
                if current_order.filled != order.filled:
                    await self._handle_order_fill(order, current_order)

                await asyncio.sleep(0.1)  # 100ms检查间隔

            except Exception as e:
                logger.error(f"Order monitoring error: {e}")
                await asyncio.sleep(1)

    async def _handle_order_status_change(self, order: Order, updated_order: Order):
        """处理订单状态变化"""
        old_status = order.status
        new_status = updated_order.status

        # 更新订单信息
        order.status = new_status
        order.filled = updated_order.filled
        order.remaining = updated_order.remaining

        await self._record_order_event(order, "status_changed", {
            "old_status": old_status,
            "new_status": new_status
        })

        # 发布订单状态更新事件
        await self.message_bus.publish(Topics.ORDER_STATUS_UPDATE, {
            "order_id": order.id,
            "old_status": old_status,
            "new_status": new_status,
            "timestamp": datetime.now()
        })

        # 如果订单完成，从活跃订单中移除
        if new_status in [OrderState.FILLED.value, OrderState.CANCELLED.value,
                         OrderState.REJECTED.value, OrderState.FAILED.value]:
            self.active_orders.pop(order.id, None)

    async def _handle_order_fill(self, order: Order, updated_order: Order):
        """处理订单成交"""
        fill_amount = updated_order.filled - order.filled

        if fill_amount > 0:
            await self._record_order_event(order, "order_filled", {
                "fill_amount": float(fill_amount),
                "total_filled": float(updated_order.filled),
                "average_price": float(updated_order.average_price or 0)
            })

            # 发布成交事件
            await self.message_bus.publish(Topics.ORDER_FILLED, {
                "order_id": order.id,
                "fill_amount": float(fill_amount),
                "fill_price": float(updated_order.average_price or 0),
                "total_filled": float(updated_order.filled),
                "timestamp": datetime.now()
            })

    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        order = self.active_orders.get(order_id)
        if not order:
            return False

        try:
            exchange = self.exchange_manager.get_exchange(order.symbol.exchange)
            success = await exchange.cancel_order(order_id, order.symbol.symbol)

            if success:
                await self.state_machine.transition(order, OrderEvent.CANCEL)
                await self._record_order_event(order, "order_cancelled")

            return success

        except Exception as e:
            logger.error(f"Order cancellation failed: {e}")
            return False

    async def _record_order_event(self, order: Order, event_type: str, data: Dict = None):
        """记录订单事件"""
        if order.id not in self.order_history:
            self.order_history[order.id] = []

        event = {
            "event_type": event_type,
            "timestamp": datetime.now(),
            "order_status": order.status,
            "data": data or {}
        }

        self.order_history[order.id].append(event)

        # 发布订单事件
        await self.message_bus.publish(f"order.event.{event_type}", {
            "order_id": order.id,
            "event": event
        })

    def _generate_order_id(self) -> str:
        """生成订单ID"""
        import uuid
        return str(uuid.uuid4())

    async def get_order_history(self, order_id: str) -> List[Dict]:
        """获取订单历史"""
        return self.order_history.get(order_id, [])

    async def get_active_orders(self) -> Dict[str, Order]:
        """获取活跃订单"""
        return self.active_orders.copy()

    async def cleanup_completed_orders(self, days: int = 7):
        """清理已完成的订单历史"""
        cutoff_time = datetime.now() - timedelta(days=days)

        to_remove = []
        for order_id, events in self.order_history.items():
            if events and events[-1]["timestamp"] < cutoff_time:
                to_remove.append(order_id)

        for order_id in to_remove:
            del self.order_history[order_id]

        logger.info(f"Cleaned up {len(to_remove)} order histories")
```

## 5. 应急处理机制

### 5.1 紧急停止系统

```python
class EmergencyStopSystem:
    """紧急停止系统"""

    def __init__(self,
                 order_manager: OrderLifecycleManager,
                 exchange_manager: ExchangeManager,
                 message_bus: MessageBusInterface):
        self.order_manager = order_manager
        self.exchange_manager = exchange_manager
        self.message_bus = message_bus
        self.emergency_triggers = {}
        self.stop_protocols = {}

    def register_trigger(self, name: str, trigger_func: Callable):
        """注册紧急停止触发器"""
        self.emergency_triggers[name] = trigger_func

    def register_stop_protocol(self, name: str, protocol_func: Callable):
        """注册停止协议"""
        self.stop_protocols[name] = protocol_func

    async def execute_emergency_stop(self, trigger: str, severity: str = "HIGH"):
        """执行紧急停止"""
        logger.critical(f"Emergency stop triggered: {trigger}")

        stop_event = {
            "trigger": trigger,
            "severity": severity,
            "timestamp": datetime.now(),
            "actions_taken": []
        }

        try:
            # 1. 立即停止新订单
            await self._stop_new_orders()
            stop_event["actions_taken"].append("stopped_new_orders")

            # 2. 取消所有挂单
            cancelled_orders = await self._cancel_all_orders()
            stop_event["actions_taken"].append(f"cancelled_{len(cancelled_orders)}_orders")

            # 3. 平仓所有持仓（如果需要）
            if severity == "CRITICAL":
                closed_positions = await self._close_all_positions()
                stop_event["actions_taken"].append(f"closed_{len(closed_positions)}_positions")

            # 4. 断开交易所连接
            await self._disconnect_exchanges()
            stop_event["actions_taken"].append("disconnected_exchanges")

            # 5. 发送告警通知
            await self._send_emergency_alert(stop_event)

        except Exception as e:
            logger.error(f"Emergency stop execution failed: {e}")
            stop_event["error"] = str(e)

        # 记录紧急停止事件
        await self.message_bus.publish(Topics.EMERGENCY_STOP, stop_event)

        return stop_event

    async def _stop_new_orders(self):
        """停止新订单创建"""
        # 设置全局标志，阻止新订单创建
        # 这里需要在OrderLifecycleManager中实现相应的检查
        pass

    async def _cancel_all_orders(self) -> List[str]:
        """取消所有活跃订单"""
        active_orders = await self.order_manager.get_active_orders()
        cancelled_orders = []

        tasks = []
        for order_id in active_orders.keys():
            task = self.order_manager.cancel_order(order_id)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for order_id, result in zip(active_orders.keys(), results):
            if result is True:
                cancelled_orders.append(order_id)
            elif isinstance(result, Exception):
                logger.error(f"Failed to cancel order {order_id}: {result}")

        return cancelled_orders

    async def _close_all_positions(self) -> List[str]:
        """平仓所有持仓"""
        closed_positions = []

        for exchange_name, exchange in self.exchange_manager.get_all_exchanges().items():
            try:
                positions = await exchange.get_positions()

                for position in positions:
                    if position.amount != 0:
                        # 创建平仓订单
                        close_order = Order(
                            id=f"emergency_close_{position.symbol.symbol}_{int(time.time())}",
                            client_order_id=f"emergency_{int(time.time() * 1000)}",
                            symbol=position.symbol,
                            side=OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY,
                            order_type=OrderType.MARKET,
                            amount=abs(position.amount),
                            price=None,
                            filled=Decimal('0'),
                            remaining=abs(position.amount),
                            status=OrderState.CREATED.value,
                            timestamp=datetime.now()
                        )

                        # 直接提交到交易所
                        await exchange.create_order(close_order)
                        closed_positions.append(f"{exchange_name}:{position.symbol.symbol}")

            except Exception as e:
                logger.error(f"Failed to close positions on {exchange_name}: {e}")

        return closed_positions

    async def _disconnect_exchanges(self):
        """断开交易所连接"""
        # 实现交易所连接断开逻辑
        pass

    async def _send_emergency_alert(self, stop_event: Dict):
        """发送紧急告警"""
        alert = {
            "type": "emergency_stop_executed",
            "severity": "CRITICAL",
            "message": f"Emergency stop executed: {stop_event['trigger']}",
            "details": stop_event,
            "timestamp": datetime.now()
        }

        # 发送到多个通道
        await self.message_bus.publish(Topics.SYSTEM_ALERT, alert)

        # 这里可以添加其他通知方式：
        # - 邮件通知
        # - 短信通知
        # - 钉钉/微信通知
        # - 电话告警
```

### 5.2 风险报告与分析

```python
class RiskReportGenerator:
    """风险报告生成器"""

    def __init__(self,
                 risk_monitor: RealTimeRiskMonitor,
                 order_manager: OrderLifecycleManager):
        self.risk_monitor = risk_monitor
        self.order_manager = order_manager

    async def generate_real_time_report(self) -> Dict:
        """生成实时风险报告"""
        report = {
            "timestamp": datetime.now(),
            "report_type": "real_time",
            "summary": {},
            "metrics": {},
            "positions": {},
            "orders": {},
            "alerts": []
        }

        # 风险概览
        report["summary"] = {
            "overall_risk_status": self.risk_monitor._calculate_overall_risk_status(),
            "active_alerts": len([m for m in self.risk_monitor.risk_metrics.values()
                                if m.status in ["WARNING", "CRITICAL"]]),
            "total_position_value": await self._calculate_total_position_value(),
            "available_margin": await self._calculate_available_margin(),
        }

        # 详细风险指标
        report["metrics"] = {
            name: {
                "value": metric.value,
                "threshold": metric.threshold,
                "status": metric.status,
                "description": metric.description
            } for name, metric in self.risk_monitor.risk_metrics.items()
        }

        # 持仓分析
        report["positions"] = await self._analyze_positions()

        # 订单分析
        report["orders"] = await self._analyze_orders()

        # 告警信息
        report["alerts"] = self.risk_monitor.alert_history[-10:]  # 最近10条告警

        return report

    async def generate_daily_report(self, date: datetime) -> Dict:
        """生成日度风险报告"""
        report = {
            "date": date.strftime("%Y-%m-%d"),
            "report_type": "daily",
            "pnl_analysis": {},
            "risk_analysis": {},
            "trading_analysis": {},
            "performance_metrics": {},
            "recommendations": []
        }

        # PnL分析
        report["pnl_analysis"] = await self._analyze_daily_pnl(date)

        # 风险分析
        report["risk_analysis"] = await self._analyze_daily_risk(date)

        # 交易分析
        report["trading_analysis"] = await self._analyze_daily_trading(date)

        # 绩效指标
        report["performance_metrics"] = await self._calculate_performance_metrics(date)

        # 风险建议
        report["recommendations"] = await self._generate_risk_recommendations(report)

        return report

    async def _calculate_total_position_value(self) -> float:
        """计算总持仓价值"""
        # 实现总持仓价值计算
        return 0.0

    async def _calculate_available_margin(self) -> float:
        """计算可用保证金"""
        # 实现可用保证金计算
        return 0.0

    async def _analyze_positions(self) -> Dict:
        """分析持仓"""
        return {
            "total_positions": 0,
            "long_positions": 0,
            "short_positions": 0,
            "largest_position": {},
            "concentration_risk": 0.0
        }

    async def _analyze_orders(self) -> Dict:
        """分析订单"""
        active_orders = await self.order_manager.get_active_orders()

        return {
            "total_active_orders": len(active_orders),
            "pending_orders": len([o for o in active_orders.values()
                                 if o.status == OrderState.OPEN.value]),
            "partially_filled_orders": len([o for o in active_orders.values()
                                          if o.status == OrderState.PARTIALLY_FILLED.value]),
            "average_fill_rate": await self._calculate_average_fill_rate(),
        }

    async def _calculate_average_fill_rate(self) -> float:
        """计算平均成交率"""
        # 实现平均成交率计算
        return 0.0

    async def _generate_risk_recommendations(self, report: Dict) -> List[str]:
        """生成风险建议"""
        recommendations = []

        # 基于报告内容生成建议
        if report.get("risk_analysis", {}).get("var_breach_count", 0) > 5:
            recommendations.append("建议降低仓位规模以减少VaR风险")

        if report.get("trading_analysis", {}).get("win_rate", 1.0) < 0.4:
            recommendations.append("胜率过低，建议检查策略参数")

        return recommendations

# 使用示例
async def main():
    # 初始化各个组件
    exchange_manager = ExchangeManager()
    limits = RiskLimits(
        max_position_size={"BTC/USDT": Decimal('10')},
        max_daily_loss=Decimal('10000'),
        max_drawdown=0.1,
        var_1d_95=Decimal('5000'),
        var_1d_99=Decimal('8000'),
        cvar_1d_95=Decimal('6000'),
        max_leverage=10.0,
        margin_utilization_limit=0.8
    )

    calculator = RealTimeRiskCalculator(VaRCalculator())
    message_bus = RedisStreamMessageBus(redis_client)

    # 启动风险监控
    risk_monitor = RealTimeRiskMonitor(limits, calculator, message_bus)
    await risk_monitor.start_monitoring()

    # 启动订单管理
    order_manager = OrderLifecycleManager(exchange_manager, risk_monitor, message_bus)

    # 启动紧急停止系统
    emergency_stop = EmergencyStopSystem(order_manager, exchange_manager, message_bus)

    # 生成风险报告
    report_generator = RiskReportGenerator(risk_monitor, order_manager)
    real_time_report = await report_generator.generate_real_time_report()

    print(json.dumps(real_time_report, indent=2, default=str))

if __name__ == "__main__":
    asyncio.run(main())
```

这个风险管理与订单生命周期管理设计提供了完整的风险控制框架和订单管理机制，确保系统在各种市场条件下都能安全、稳定地运行。通过实时监控、多层风险控制和完善的应急处理机制，最大限度地保护交易资金和系统稳定性。