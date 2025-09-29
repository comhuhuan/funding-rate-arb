# Funding Rate Arbitrage System

A production-grade cryptocurrency funding rate arbitrage trading system built with Python.

## 🚀 Features

- **Multi-Exchange Support**: Pluggable architecture supporting Binance, OKX, Bybit, and custom exchanges
- **Real-time Arbitrage**: Millisecond-level opportunity detection and execution
- **Advanced Risk Management**: Comprehensive risk controls with VaR calculation and position limits
- **High Performance**: Async architecture with Redis caching and message queues
- **Production Ready**: Complete monitoring, logging, and deployment solutions

## 🏗️ Architecture

The system follows a modular, event-driven architecture:

- **Exchange Layer**: Unified interface for multiple exchanges via CCXT and custom adapters
- **Data Layer**: Real-time market data processing with multi-tier caching
- **Strategy Layer**: Pluggable arbitrage strategies with risk management
- **Execution Layer**: Order lifecycle management with smart routing
- **Infrastructure Layer**: Redis, PostgreSQL, InfluxDB for caching, data, and metrics

## 📋 Prerequisites

- Python 3.9+
- Redis 6.0+
- PostgreSQL 13+
- InfluxDB 1.8+ (for time series data)

## 🛠️ Installation

### Development Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd funding-rate-arb
```

2. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements-dev.txt
```

4. Copy configuration template:
```bash
cp config/main.yaml.template config/main.yaml
```

5. Configure your exchange API credentials in `config/exchanges.yaml`

### Production Setup

See [deployment documentation](docs/CONFIGURATION_DEPLOYMENT.md) for production deployment with Docker and Kubernetes.

## 🧪 Testing

Run the test suite:

```bash
# Unit tests
pytest tests/unit/

# Integration tests
pytest tests/integration/

# All tests with coverage
pytest --cov=src tests/
```

## 📊 Monitoring

The system includes comprehensive monitoring with:

- **Prometheus**: Metrics collection
- **Grafana**: Dashboards and visualization
- **Custom alerts**: Risk and performance monitoring

Access the monitoring dashboard at `http://localhost:3000` (Grafana) after deployment.

## 🔧 Configuration

### Exchange Configuration

Configure exchanges in `config/exchanges.yaml`:

```yaml
exchanges:
  binance:
    enabled: true
    api_key: "${BINANCE_API_KEY}"
    secret_key: "${BINANCE_SECRET_KEY}"
    sandbox: false
```

### Strategy Configuration

Configure strategies in `config/strategies.yaml`:

```yaml
strategies:
  funding_rate_arbitrage:
    enabled: true
    symbols: ["BTC/USDT", "ETH/USDT"]
    parameters:
      min_spread: 0.0001
      position_size: 1000
```

### Risk Management

Configure risk limits in `config/risk.yaml`:

```yaml
risk_management:
  global_limits:
    max_daily_loss: 5000
    max_drawdown: 0.1
    max_position_size: 10000
```

## 🚦 Usage

### Start the System

```bash
# Development mode
python main.py

# Production mode with Docker
docker-compose up -d
```

### Monitor System Status

```bash
# Check system health
curl http://localhost:8000/health

# View metrics
curl http://localhost:9090/metrics
```

## 📖 Documentation

- [System Architecture](docs/SYSTEM_ARCHITECTURE.md)
- [Module Design](docs/MODULE_DESIGN.md)
- [Data Flow Design](docs/DATA_FLOW_DESIGN.md)
- [Risk Management](docs/RISK_ORDER_MANAGEMENT.md)
- [Exchange Plugins](docs/EXCHANGE_PLUGIN_ARCHITECTURE.md)
- [Configuration & Deployment](docs/CONFIGURATION_DEPLOYMENT.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Write tests for your changes
4. Implement your changes
5. Run the test suite: `pytest`
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

## ⚠️ Risk Disclaimer

This is trading software that involves financial risk.

**IMPORTANT**:
- Always test strategies in sandbox/testnet environments first
- Never risk more than you can afford to lose
- Past performance does not guarantee future results
- This software is provided "as is" without warranty

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- 📚 [Documentation](docs/)
- 🐛 [Issue Tracker](https://github.com/your-username/funding-rate-arb/issues)
- 💬 [Discussions](https://github.com/your-username/funding-rate-arb/discussions)

## 🏗️ Development Status

- [x] System Architecture Design
- [x] Core Module Interfaces
- [ ] Exchange Abstraction Layer
- [ ] Market Data Processing
- [ ] Risk Management System
- [ ] Order Management System
- [ ] Strategy Framework
- [ ] Production Deployment

## 🔬 Research & Backtesting

For research and backtesting capabilities, see the `research/` directory with Jupyter notebooks for:

- Historical funding rate analysis
- Strategy backtesting
- Performance attribution
- Risk analysis