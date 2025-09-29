"""
Core data types and models for the funding rate arbitrage system.

This module defines all the fundamental data structures used throughout
the trading system, including symbols, orders, positions, and arbitrage
opportunities.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Any, Union


# =============================================================================
# Enums
# =============================================================================

class ExchangeType(Enum):
    """Supported exchange types."""
    BINANCE = "binance"
    OKX = "okx"
    BYBIT = "bybit"
    HUOBI = "huobi"
    CUSTOM = "custom"


class OrderSide(Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill


class OrderStatus(Enum):
    """Order status enumeration."""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class ArbitrageType(Enum):
    """Types of arbitrage strategies."""
    SPOT_FUTURES = "spot_futures"
    FUTURES_FUTURES = "futures_futures"
    CROSS_EXCHANGE = "cross_exchange"
    FUNDING_RATE = "funding_rate"


class Priority(Enum):
    """Priority levels for trading signals and opportunities."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class Symbol:
    """
    Trading symbol representation.

    Attributes:
        base: Base currency (e.g., 'BTC')
        quote: Quote currency (e.g., 'USDT')
        exchange: Exchange name (e.g., 'binance')
        symbol: Full symbol string (e.g., 'BTC/USDT')
        contract_type: Optional contract type ('spot', 'futures', 'perpetual')
    """
    base: str
    quote: str
    exchange: str
    symbol: str
    contract_type: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate symbol format."""
        if "/" not in self.symbol:
            raise ValueError(f"Invalid symbol format: {self.symbol}")

        parts = self.symbol.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid symbol format: {self.symbol}")

        # Validate that base/quote match the symbol
        symbol_base, symbol_quote = parts
        if symbol_base != self.base or symbol_quote != self.quote:
            raise ValueError(
                f"Symbol parts mismatch: {self.base}/{self.quote} != {self.symbol}"
            )

    @classmethod
    def from_string(cls, symbol_str: str, exchange: str,
                   contract_type: Optional[str] = None) -> Symbol:
        """
        Create Symbol from string representation.

        Args:
            symbol_str: Symbol string like 'BTC/USDT'
            exchange: Exchange name
            contract_type: Optional contract type

        Returns:
            Symbol instance

        Raises:
            ValueError: If symbol format is invalid
        """
        if "/" not in symbol_str:
            raise ValueError(f"Invalid symbol format: {symbol_str}")

        base, quote = symbol_str.split("/", 1)
        return cls(
            base=base,
            quote=quote,
            exchange=exchange,
            symbol=symbol_str,
            contract_type=contract_type
        )

    def __str__(self) -> str:
        """String representation."""
        return f"{self.exchange}:{self.symbol}"


@dataclass
class Price:
    """
    Price level in order book.

    Attributes:
        price: Price value
        volume: Volume at this price level
        timestamp: When this price was recorded
    """
    price: Decimal
    volume: Decimal
    timestamp: datetime

    def __post_init__(self) -> None:
        """Convert numeric types to Decimal."""
        if not isinstance(self.price, Decimal):
            self.price = Decimal(str(self.price))
        if not isinstance(self.volume, Decimal):
            self.volume = Decimal(str(self.volume))


@dataclass
class OrderBook:
    """
    Order book representation.

    Attributes:
        symbol: Trading symbol
        bids: List of bid prices (sorted highest to lowest)
        asks: List of ask prices (sorted lowest to highest)
        timestamp: Order book timestamp
        exchange: Exchange name
    """
    symbol: Symbol
    bids: List[Price]
    asks: List[Price]
    timestamp: datetime
    exchange: str

    @property
    def best_bid(self) -> Optional[Decimal]:
        """Get the best (highest) bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        """Get the best (lowest) ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Optional[Decimal]:
        """Get the bid-ask spread."""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def mid_price(self) -> Optional[Decimal]:
        """Get the mid price between best bid and ask."""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / Decimal("2")
        return None


@dataclass
class Trade:
    """
    Trade execution record.

    Attributes:
        id: Trade ID
        symbol: Trading symbol
        side: Order side (buy/sell)
        amount: Trade amount
        price: Execution price
        cost: Total cost (amount * price)
        fee: Trading fee
        timestamp: Execution timestamp
        exchange: Exchange name
    """
    id: str
    symbol: Symbol
    side: OrderSide
    amount: Decimal
    price: Decimal
    cost: Decimal
    fee: Decimal
    timestamp: datetime
    exchange: str

    def __post_init__(self) -> None:
        """Convert numeric types to Decimal."""
        for field_name in ["amount", "price", "cost", "fee"]:
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                setattr(self, field_name, Decimal(str(value)))


@dataclass
class FundingRate:
    """
    Funding rate information for perpetual contracts.

    Attributes:
        exchange: Exchange name
        symbol: Trading symbol
        rate: Current funding rate (as decimal, e.g., 0.0001 = 0.01%)
        timestamp: When the rate was recorded
        next_funding_time: Next funding settlement time
        funding_interval: Funding interval in seconds
        predicted_rate: Predicted next funding rate (optional)
    """
    exchange: str
    symbol: str
    rate: Decimal
    timestamp: datetime
    next_funding_time: datetime
    funding_interval: int
    predicted_rate: Optional[Decimal] = None

    def __post_init__(self) -> None:
        """Convert numeric types to Decimal."""
        if not isinstance(self.rate, Decimal):
            self.rate = Decimal(str(self.rate))
        if self.predicted_rate is not None and not isinstance(self.predicted_rate, Decimal):
            self.predicted_rate = Decimal(str(self.predicted_rate))

    @property
    def annualized_rate(self) -> Decimal:
        """Calculate annualized funding rate."""
        # Number of funding periods per year
        periods_per_year = (365 * 24 * 3600) / self.funding_interval
        return self.rate * Decimal(str(periods_per_year))

    @property
    def rate_percent(self) -> Decimal:
        """Get rate as percentage (rate * 100)."""
        return self.rate * Decimal("100")


@dataclass
class Position:
    """
    Trading position representation.

    Attributes:
        symbol: Trading symbol
        side: Position side (buy = long, sell = short)
        amount: Position size
        entry_price: Average entry price
        mark_price: Current mark price
        unrealized_pnl: Unrealized profit/loss
        margin: Margin used
        leverage: Leverage ratio
        timestamp: Position timestamp
    """
    symbol: Symbol
    side: OrderSide
    amount: Decimal
    entry_price: Decimal
    mark_price: Decimal
    unrealized_pnl: Decimal
    margin: Decimal
    leverage: float
    timestamp: datetime

    def __post_init__(self) -> None:
        """Convert numeric types to Decimal."""
        for field_name in ["amount", "entry_price", "mark_price", "unrealized_pnl", "margin"]:
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                setattr(self, field_name, Decimal(str(value)))

    @property
    def notional_value(self) -> Decimal:
        """Calculate notional value of position."""
        return abs(self.amount) * self.mark_price

    @property
    def pnl_percent(self) -> Decimal:
        """Calculate PnL as percentage of entry value."""
        entry_value = abs(self.amount) * self.entry_price
        if entry_value == 0:
            return Decimal("0")
        return (self.unrealized_pnl / entry_value) * Decimal("100")


@dataclass
class Balance:
    """
    Account balance for a specific currency.

    Attributes:
        currency: Currency symbol (e.g., 'BTC', 'USDT')
        free: Available balance
        used: Used/frozen balance
        total: Total balance (free + used)
        timestamp: Balance timestamp
    """
    currency: str
    free: Decimal
    used: Decimal
    total: Decimal
    timestamp: datetime

    def __post_init__(self) -> None:
        """Convert numeric types to Decimal and validate."""
        for field_name in ["free", "used", "total"]:
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                setattr(self, field_name, Decimal(str(value)))

        # Validate balance consistency
        calculated_total = self.free + self.used
        if abs(calculated_total - self.total) > Decimal("0.00000001"):
            # Allow for small rounding errors
            self.total = calculated_total


@dataclass
class Order:
    """
    Order representation.

    Attributes:
        id: Order ID from exchange
        client_order_id: Client-side order ID
        symbol: Trading symbol
        side: Order side (buy/sell)
        order_type: Order type (market/limit/etc.)
        amount: Order amount
        price: Order price (None for market orders)
        filled: Filled amount
        remaining: Remaining amount
        status: Order status
        timestamp: Order creation timestamp
        fee: Trading fee (optional)
        average_price: Average fill price (optional)
    """
    id: str
    client_order_id: str
    symbol: Symbol
    side: OrderSide
    order_type: OrderType
    amount: Decimal
    filled: Decimal
    remaining: Decimal
    status: OrderStatus
    timestamp: datetime
    price: Optional[Decimal] = None
    fee: Optional[Decimal] = None
    average_price: Optional[Decimal] = None

    def __post_init__(self) -> None:
        """Convert numeric types to Decimal."""
        for field_name in ["amount", "filled", "remaining"]:
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                setattr(self, field_name, Decimal(str(value)))

        for field_name in ["price", "fee", "average_price"]:
            value = getattr(self, field_name)
            if value is not None and not isinstance(value, Decimal):
                setattr(self, field_name, Decimal(str(value)))

    @property
    def fill_percentage(self) -> Decimal:
        """Calculate fill percentage (0.0 to 1.0)."""
        if self.amount == 0:
            return Decimal("0")
        return self.filled / self.amount

    @property
    def is_complete(self) -> bool:
        """Check if order is completely filled."""
        return self.status in [OrderStatus.FILLED] or self.remaining == 0

    @property
    def cost(self) -> Optional[Decimal]:
        """Calculate order cost based on average price or order price."""
        price = self.average_price or self.price
        if price is not None:
            return self.filled * price
        return None


@dataclass
class TradingSignal:
    """
    Trading signal from strategy.

    Attributes:
        id: Signal ID
        strategy_id: Strategy that generated the signal
        symbol: Trading symbol
        side: Order side (buy/sell)
        signal_type: Type of signal ('entry', 'exit', 'close')
        price: Suggested price (optional for market orders)
        amount: Suggested amount
        confidence: Signal confidence (0.0 to 1.0)
        urgency: Signal urgency level
        valid_until: Signal expiration time
        created_at: Signal creation time
        metadata: Additional signal metadata
    """
    id: str
    strategy_id: str
    symbol: Symbol
    side: OrderSide
    signal_type: str
    amount: Decimal
    confidence: float
    urgency: Priority
    valid_until: datetime
    created_at: datetime
    price: Optional[Decimal] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Convert numeric types to Decimal."""
        if not isinstance(self.amount, Decimal):
            self.amount = Decimal(str(self.amount))
        if self.price is not None and not isinstance(self.price, Decimal):
            self.price = Decimal(str(self.price))

    @property
    def is_expired(self) -> bool:
        """Check if signal has expired."""
        return datetime.now(timezone.utc) > self.valid_until

    @property
    def time_to_expiry(self) -> float:
        """Get time to expiry in seconds."""
        now = datetime.now(timezone.utc)
        if self.valid_until.tzinfo is None:
            # Assume UTC if no timezone info
            valid_until = self.valid_until.replace(tzinfo=timezone.utc)
        else:
            valid_until = self.valid_until

        delta = valid_until - now
        return delta.total_seconds()


@dataclass
class ArbitrageOpportunity:
    """
    Arbitrage opportunity representation.

    Attributes:
        id: Opportunity ID
        type: Type of arbitrage
        symbol: Trading symbol
        exchanges: List of exchanges involved
        entry_signals: List of trading signals for entry
        expected_profit: Expected profit in base currency
        max_profit: Maximum potential profit
        risk_score: Risk score (0.0 to 1.0, higher = riskier)
        max_position_size: Maximum recommended position size
        liquidity_score: Liquidity score (0.0 to 1.0, higher = more liquid)
        urgency: Opportunity urgency level
        valid_until: Opportunity expiration time
        created_at: Opportunity creation time
        metadata: Additional opportunity metadata
    """
    id: str
    type: ArbitrageType
    symbol: str
    exchanges: List[str]
    entry_signals: List[TradingSignal]
    expected_profit: Decimal
    max_profit: Decimal
    risk_score: float
    max_position_size: Decimal
    liquidity_score: float
    urgency: Priority
    valid_until: datetime
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Convert numeric types to Decimal."""
        for field_name in ["expected_profit", "max_profit", "max_position_size"]:
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                setattr(self, field_name, Decimal(str(value)))

    @property
    def profit_ratio(self) -> Decimal:
        """Calculate profit ratio (expected profit / max position size)."""
        if self.max_position_size == 0:
            return Decimal("0")
        return self.expected_profit / self.max_position_size

    @property
    def risk_adjusted_return(self) -> Decimal:
        """Calculate risk-adjusted return (profit ratio / risk score)."""
        if self.risk_score == 0:
            return Decimal("0")
        return self.profit_ratio / Decimal(str(self.risk_score))

    @property
    def is_expired(self) -> bool:
        """Check if opportunity has expired."""
        return datetime.now(timezone.utc) > self.valid_until

    @property
    def time_to_expiry(self) -> float:
        """Get time to expiry in seconds."""
        now = datetime.now(timezone.utc)
        if self.valid_until.tzinfo is None:
            # Assume UTC if no timezone info
            valid_until = self.valid_until.replace(tzinfo=timezone.utc)
        else:
            valid_until = self.valid_until

        delta = valid_until - now
        return delta.total_seconds()

    @property
    def quality_score(self) -> float:
        """
        Calculate overall opportunity quality score.

        Combines profit ratio, liquidity score, and inverse risk score.
        """
        profit_component = float(self.profit_ratio) * 100  # Convert to percentage
        liquidity_component = self.liquidity_score
        risk_component = 1.0 - self.risk_score  # Inverse risk (lower risk = higher score)

        # Weighted average
        weights = [0.4, 0.3, 0.3]  # Profit, liquidity, risk
        components = [profit_component, liquidity_component, risk_component]

        return sum(w * c for w, c in zip(weights, components)) / sum(weights)


# =============================================================================
# Type Aliases
# =============================================================================

# Numeric types that can be converted to Decimal
NumericType = Union[int, float, str, Decimal]

# Timestamp types
TimestampType = Union[datetime, int, float]

# Generic data dictionary
DataDict = Dict[str, Any]