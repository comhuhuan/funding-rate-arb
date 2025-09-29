"""
Unit tests for core data types and models.
"""

import pytest
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Any

from src.types import (
    Symbol,
    Price,
    OrderBook,
    Trade,
    FundingRate,
    Position,
    Balance,
    Order,
    ArbitrageOpportunity,
    TradingSignal,
    ExchangeType,
    OrderSide,
    OrderType,
    OrderStatus,
    ArbitrageType,
    Priority,
)


class TestExchangeType:
    """Test ExchangeType enum."""

    def test_exchange_type_values(self):
        """Test ExchangeType enum values."""
        assert ExchangeType.BINANCE.value == "binance"
        assert ExchangeType.OKX.value == "okx"
        assert ExchangeType.BYBIT.value == "bybit"
        assert ExchangeType.HUOBI.value == "huobi"
        assert ExchangeType.CUSTOM.value == "custom"


class TestOrderEnums:
    """Test order-related enums."""

    def test_order_side(self):
        """Test OrderSide enum."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"

    def test_order_type(self):
        """Test OrderType enum."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.IOC.value == "ioc"
        assert OrderType.FOK.value == "fok"

    def test_order_status(self):
        """Test OrderStatus enum."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.OPEN.value == "open"
        assert OrderStatus.PARTIALLY_FILLED.value == "partially_filled"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"


class TestArbitrageEnums:
    """Test arbitrage-related enums."""

    def test_arbitrage_type(self):
        """Test ArbitrageType enum."""
        assert ArbitrageType.SPOT_FUTURES.value == "spot_futures"
        assert ArbitrageType.FUTURES_FUTURES.value == "futures_futures"
        assert ArbitrageType.CROSS_EXCHANGE.value == "cross_exchange"
        assert ArbitrageType.FUNDING_RATE.value == "funding_rate"

    def test_priority(self):
        """Test Priority enum."""
        assert Priority.LOW.value == 1
        assert Priority.MEDIUM.value == 2
        assert Priority.HIGH.value == 3
        assert Priority.URGENT.value == 4


class TestSymbol:
    """Test Symbol data class."""

    def test_symbol_creation(self):
        """Test Symbol creation with valid data."""
        symbol = Symbol(
            base="BTC",
            quote="USDT",
            exchange="binance",
            symbol="BTC/USDT",
            contract_type="spot"
        )

        assert symbol.base == "BTC"
        assert symbol.quote == "USDT"
        assert symbol.exchange == "binance"
        assert symbol.symbol == "BTC/USDT"
        assert symbol.contract_type == "spot"

    def test_symbol_defaults(self):
        """Test Symbol with default contract_type."""
        symbol = Symbol(
            base="ETH",
            quote="USDT",
            exchange="okx",
            symbol="ETH/USDT"
        )

        assert symbol.contract_type is None

    def test_symbol_from_string(self):
        """Test Symbol.from_string class method."""
        symbol = Symbol.from_string("BTC/USDT", "binance")

        assert symbol.base == "BTC"
        assert symbol.quote == "USDT"
        assert symbol.exchange == "binance"
        assert symbol.symbol == "BTC/USDT"

    def test_symbol_from_string_invalid(self):
        """Test Symbol.from_string with invalid symbol string."""
        with pytest.raises(ValueError, match="Invalid symbol format"):
            Symbol.from_string("BTCUSDT", "binance")

    def test_symbol_equality(self):
        """Test Symbol equality comparison."""
        symbol1 = Symbol("BTC", "USDT", "binance", "BTC/USDT")
        symbol2 = Symbol("BTC", "USDT", "binance", "BTC/USDT")
        symbol3 = Symbol("ETH", "USDT", "binance", "ETH/USDT")

        assert symbol1 == symbol2
        assert symbol1 != symbol3

    def test_symbol_hash(self):
        """Test Symbol can be used as dict key."""
        symbol1 = Symbol("BTC", "USDT", "binance", "BTC/USDT")
        symbol2 = Symbol("BTC", "USDT", "binance", "BTC/USDT")

        symbol_dict = {symbol1: "value1"}
        symbol_dict[symbol2] = "value2"

        # Should have only one key since symbols are equal
        assert len(symbol_dict) == 1
        assert symbol_dict[symbol1] == "value2"


class TestPrice:
    """Test Price data class."""

    def test_price_creation(self):
        """Test Price creation with valid data."""
        timestamp = datetime.now(timezone.utc)
        price = Price(
            price=Decimal("50000.50"),
            volume=Decimal("1.5"),
            timestamp=timestamp
        )

        assert price.price == Decimal("50000.50")
        assert price.volume == Decimal("1.5")
        assert price.timestamp == timestamp

    def test_price_decimal_conversion(self):
        """Test Price with float inputs are converted to Decimal."""
        price = Price(
            price=50000.50,
            volume=1.5,
            timestamp=datetime.now(timezone.utc)
        )

        assert isinstance(price.price, Decimal)
        assert isinstance(price.volume, Decimal)
        assert price.price == Decimal("50000.50")
        assert price.volume == Decimal("1.5")


class TestOrderBook:
    """Test OrderBook data class."""

    def test_orderbook_creation(self):
        """Test OrderBook creation with valid data."""
        symbol = Symbol("BTC", "USDT", "binance", "BTC/USDT")
        timestamp = datetime.now(timezone.utc)

        bids = [
            Price(Decimal("50000"), Decimal("1.0"), timestamp),
            Price(Decimal("49999"), Decimal("0.5"), timestamp),
        ]
        asks = [
            Price(Decimal("50001"), Decimal("0.8"), timestamp),
            Price(Decimal("50002"), Decimal("1.2"), timestamp),
        ]

        orderbook = OrderBook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=timestamp,
            exchange="binance"
        )

        assert orderbook.symbol == symbol
        assert len(orderbook.bids) == 2
        assert len(orderbook.asks) == 2
        assert orderbook.timestamp == timestamp
        assert orderbook.exchange == "binance"

    def test_orderbook_best_bid_ask(self):
        """Test OrderBook best bid/ask properties."""
        symbol = Symbol("BTC", "USDT", "binance", "BTC/USDT")
        timestamp = datetime.now(timezone.utc)

        bids = [
            Price(Decimal("50000"), Decimal("1.0"), timestamp),
            Price(Decimal("49999"), Decimal("0.5"), timestamp),
        ]
        asks = [
            Price(Decimal("50001"), Decimal("0.8"), timestamp),
            Price(Decimal("50002"), Decimal("1.2"), timestamp),
        ]

        orderbook = OrderBook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=timestamp,
            exchange="binance"
        )

        assert orderbook.best_bid == Decimal("50000")
        assert orderbook.best_ask == Decimal("50001")
        assert orderbook.spread == Decimal("1")
        assert orderbook.mid_price == Decimal("50000.5")

    def test_orderbook_empty_bids_asks(self):
        """Test OrderBook with empty bids/asks."""
        symbol = Symbol("BTC", "USDT", "binance", "BTC/USDT")
        timestamp = datetime.now(timezone.utc)

        orderbook = OrderBook(
            symbol=symbol,
            bids=[],
            asks=[],
            timestamp=timestamp,
            exchange="binance"
        )

        assert orderbook.best_bid is None
        assert orderbook.best_ask is None
        assert orderbook.spread is None
        assert orderbook.mid_price is None


class TestFundingRate:
    """Test FundingRate data class."""

    def test_funding_rate_creation(self):
        """Test FundingRate creation with valid data."""
        timestamp = datetime.now(timezone.utc)
        next_funding_time = datetime.now(timezone.utc)

        funding_rate = FundingRate(
            exchange="binance",
            symbol="BTC/USDT",
            rate=Decimal("0.0001"),
            timestamp=timestamp,
            next_funding_time=next_funding_time,
            funding_interval=28800,  # 8 hours
            predicted_rate=Decimal("0.00015")
        )

        assert funding_rate.exchange == "binance"
        assert funding_rate.symbol == "BTC/USDT"
        assert funding_rate.rate == Decimal("0.0001")
        assert funding_rate.timestamp == timestamp
        assert funding_rate.next_funding_time == next_funding_time
        assert funding_rate.funding_interval == 28800
        assert funding_rate.predicted_rate == Decimal("0.00015")

    def test_funding_rate_optional_predicted_rate(self):
        """Test FundingRate with optional predicted_rate."""
        funding_rate = FundingRate(
            exchange="okx",
            symbol="ETH/USDT",
            rate=Decimal("0.0002"),
            timestamp=datetime.now(timezone.utc),
            next_funding_time=datetime.now(timezone.utc),
            funding_interval=28800
        )

        assert funding_rate.predicted_rate is None

    def test_funding_rate_annualized_rate(self):
        """Test FundingRate annualized_rate property."""
        funding_rate = FundingRate(
            exchange="binance",
            symbol="BTC/USDT",
            rate=Decimal("0.0001"),  # 0.01% per 8 hours
            timestamp=datetime.now(timezone.utc),
            next_funding_time=datetime.now(timezone.utc),
            funding_interval=28800  # 8 hours
        )

        # 0.01% per 8 hours = 0.01% * 3 * 365 = 10.95% annually
        expected_annual = Decimal("0.0001") * 3 * 365
        assert funding_rate.annualized_rate == expected_annual


class TestOrder:
    """Test Order data class."""

    def test_order_creation(self):
        """Test Order creation with valid data."""
        symbol = Symbol("BTC", "USDT", "binance", "BTC/USDT")
        timestamp = datetime.now(timezone.utc)

        order = Order(
            id="order_123",
            client_order_id="client_456",
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            price=Decimal("50000"),
            filled=Decimal("0.5"),
            remaining=Decimal("0.5"),
            status=OrderStatus.PARTIALLY_FILLED,
            timestamp=timestamp,
            fee=Decimal("0.1"),
            average_price=Decimal("49999")
        )

        assert order.id == "order_123"
        assert order.client_order_id == "client_456"
        assert order.symbol == symbol
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.amount == Decimal("1.0")
        assert order.price == Decimal("50000")
        assert order.filled == Decimal("0.5")
        assert order.remaining == Decimal("0.5")
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.timestamp == timestamp
        assert order.fee == Decimal("0.1")
        assert order.average_price == Decimal("49999")

    def test_order_fill_percentage(self):
        """Test Order fill_percentage property."""
        symbol = Symbol("BTC", "USDT", "binance", "BTC/USDT")

        order = Order(
            id="order_123",
            client_order_id="client_456",
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("2.0"),
            filled=Decimal("0.5"),
            remaining=Decimal("1.5"),
            status=OrderStatus.PARTIALLY_FILLED,
            timestamp=datetime.now(timezone.utc)
        )

        assert order.fill_percentage == Decimal("0.25")  # 0.5 / 2.0 = 25%

    def test_order_is_complete(self):
        """Test Order is_complete property."""
        symbol = Symbol("BTC", "USDT", "binance", "BTC/USDT")

        # Partially filled order
        partial_order = Order(
            id="order_123",
            client_order_id="client_456",
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            filled=Decimal("0.5"),
            remaining=Decimal("0.5"),
            status=OrderStatus.PARTIALLY_FILLED,
            timestamp=datetime.now(timezone.utc)
        )

        # Filled order
        filled_order = Order(
            id="order_124",
            client_order_id="client_457",
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            amount=Decimal("1.0"),
            filled=Decimal("1.0"),
            remaining=Decimal("0.0"),
            status=OrderStatus.FILLED,
            timestamp=datetime.now(timezone.utc)
        )

        assert not partial_order.is_complete
        assert filled_order.is_complete


class TestArbitrageOpportunity:
    """Test ArbitrageOpportunity data class."""

    def test_arbitrage_opportunity_creation(self):
        """Test ArbitrageOpportunity creation with valid data."""
        created_at = datetime.now(timezone.utc)
        valid_until = datetime.now(timezone.utc)

        signals = [
            TradingSignal(
                id="signal_1",
                strategy_id="funding_rate_arb",
                symbol=Symbol("BTC", "USDT", "binance", "BTC/USDT"),
                side=OrderSide.BUY,
                signal_type="entry",
                amount=Decimal("1.0"),
                confidence=0.85,
                urgency=Priority.HIGH,
                valid_until=valid_until,
                created_at=created_at
            )
        ]

        opportunity = ArbitrageOpportunity(
            id="arb_123",
            type=ArbitrageType.FUNDING_RATE,
            symbol="BTC/USDT",
            exchanges=["binance", "okx"],
            entry_signals=signals,
            expected_profit=Decimal("100.0"),
            max_profit=Decimal("150.0"),
            risk_score=0.3,
            max_position_size=Decimal("10000"),
            liquidity_score=0.9,
            urgency=Priority.HIGH,
            valid_until=valid_until,
            created_at=created_at,
            metadata={"strategy": "funding_rate_arbitrage"}
        )

        assert opportunity.id == "arb_123"
        assert opportunity.type == ArbitrageType.FUNDING_RATE
        assert opportunity.symbol == "BTC/USDT"
        assert opportunity.exchanges == ["binance", "okx"]
        assert len(opportunity.entry_signals) == 1
        assert opportunity.expected_profit == Decimal("100.0")
        assert opportunity.max_profit == Decimal("150.0")
        assert opportunity.risk_score == 0.3
        assert opportunity.max_position_size == Decimal("10000")
        assert opportunity.liquidity_score == 0.9
        assert opportunity.urgency == Priority.HIGH
        assert opportunity.valid_until == valid_until
        assert opportunity.created_at == created_at
        assert opportunity.metadata == {"strategy": "funding_rate_arbitrage"}

    def test_arbitrage_opportunity_profit_ratio(self):
        """Test ArbitrageOpportunity profit_ratio property."""
        opportunity = ArbitrageOpportunity(
            id="arb_123",
            type=ArbitrageType.FUNDING_RATE,
            symbol="BTC/USDT",
            exchanges=["binance", "okx"],
            entry_signals=[],
            expected_profit=Decimal("100.0"),
            max_profit=Decimal("200.0"),
            risk_score=0.3,
            max_position_size=Decimal("10000"),
            liquidity_score=0.9,
            urgency=Priority.HIGH,
            valid_until=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc)
        )

        # Expected profit / Max position size = 100 / 10000 = 0.01 (1%)
        assert opportunity.profit_ratio == Decimal("0.01")

    def test_arbitrage_opportunity_risk_adjusted_return(self):
        """Test ArbitrageOpportunity risk_adjusted_return property."""
        opportunity = ArbitrageOpportunity(
            id="arb_123",
            type=ArbitrageType.FUNDING_RATE,
            symbol="BTC/USDT",
            exchanges=["binance", "okx"],
            entry_signals=[],
            expected_profit=Decimal("100.0"),
            max_profit=Decimal("200.0"),
            risk_score=0.2,  # 20% risk
            max_position_size=Decimal("10000"),
            liquidity_score=0.9,
            urgency=Priority.HIGH,
            valid_until=datetime.now(timezone.utc),
            created_at=datetime.now(timezone.utc)
        )

        # (Expected profit / Max position size) / Risk score = 0.01 / 0.2 = 0.05
        expected_risk_adjusted = Decimal("0.01") / Decimal("0.2")
        assert opportunity.risk_adjusted_return == expected_risk_adjusted


class TestTradingSignal:
    """Test TradingSignal data class."""

    def test_trading_signal_creation(self):
        """Test TradingSignal creation with valid data."""
        symbol = Symbol("BTC", "USDT", "binance", "BTC/USDT")
        created_at = datetime.now(timezone.utc)
        valid_until = datetime.now(timezone.utc)

        signal = TradingSignal(
            id="signal_123",
            strategy_id="funding_rate_arb",
            symbol=symbol,
            side=OrderSide.BUY,
            signal_type="entry",
            price=Decimal("50000"),
            amount=Decimal("1.0"),
            confidence=0.85,
            urgency=Priority.HIGH,
            valid_until=valid_until,
            created_at=created_at,
            metadata={"reason": "funding_rate_positive"}
        )

        assert signal.id == "signal_123"
        assert signal.strategy_id == "funding_rate_arb"
        assert signal.symbol == symbol
        assert signal.side == OrderSide.BUY
        assert signal.signal_type == "entry"
        assert signal.price == Decimal("50000")
        assert signal.amount == Decimal("1.0")
        assert signal.confidence == 0.85
        assert signal.urgency == Priority.HIGH
        assert signal.valid_until == valid_until
        assert signal.created_at == created_at
        assert signal.metadata == {"reason": "funding_rate_positive"}

    def test_trading_signal_is_expired(self):
        """Test TradingSignal is_expired property."""
        symbol = Symbol("BTC", "USDT", "binance", "BTC/USDT")
        past_time = datetime(2023, 1, 1, tzinfo=timezone.utc)
        future_time = datetime(2026, 1, 1, tzinfo=timezone.utc)

        expired_signal = TradingSignal(
            id="signal_123",
            strategy_id="test",
            symbol=symbol,
            side=OrderSide.BUY,
            signal_type="entry",
            amount=Decimal("1.0"),
            confidence=0.8,
            urgency=Priority.MEDIUM,
            valid_until=past_time,
            created_at=past_time
        )

        valid_signal = TradingSignal(
            id="signal_124",
            strategy_id="test",
            symbol=symbol,
            side=OrderSide.BUY,
            signal_type="entry",
            amount=Decimal("1.0"),
            confidence=0.8,
            urgency=Priority.MEDIUM,
            valid_until=future_time,
            created_at=datetime.now(timezone.utc)
        )

        assert expired_signal.is_expired
        assert not valid_signal.is_expired