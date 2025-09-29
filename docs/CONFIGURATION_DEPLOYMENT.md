# 配置管理与部署方案

## 1. 配置管理体系

配置管理是系统灵活性和可维护性的关键，本系统采用分层配置架构，支持环境隔离、热更新、版本管理等功能。

```
┌─────────────────────────────────────────────────────────────┐
│                    配置管理架构                                │
├─────────────────────────────────────────────────────────────┤
│  Environment  │  Application  │  Strategy  │  Exchange      │
│   Config      │    Config     │   Config   │   Config       │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                   配置存储层                                 │
├─────────────────────────────────────────────────────────────┤
│  Local Files  │  Consul  │  Environment  │  Database       │
└─────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────┐
│                   配置管理工具                               │
├─────────────────────────────────────────────────────────────┤
│  Config Loader │ Hot Reload │ Validation │ Encryption      │
└─────────────────────────────────────────────────────────────┘
```

## 2. 配置结构设计

### 2.1 主配置文件结构

```yaml
# config/main.yaml
system:
  name: "funding-rate-arb"
  version: "1.0.0"
  environment: "development"  # development, staging, production
  debug: true
  timezone: "UTC"

logging:
  level: "INFO"
  format: "json"
  output: "stdout"
  rotation:
    max_size: "100MB"
    max_files: 10
    max_age: 30

database:
  postgresql:
    host: "${DB_HOST:localhost}"
    port: "${DB_PORT:5432}"
    database: "${DB_NAME:funding_arb}"
    username: "${DB_USER:postgres}"
    password: "${DB_PASSWORD:postgres}"
    pool_size: 20
    max_overflow: 30

  influxdb:
    host: "${INFLUX_HOST:localhost}"
    port: "${INFLUX_PORT:8086}"
    database: "${INFLUX_DB:market_data}"
    username: "${INFLUX_USER:admin}"
    password: "${INFLUX_PASSWORD:admin}"
    retention_policy: "30d"

redis:
  host: "${REDIS_HOST:localhost}"
  port: "${REDIS_PORT:6379}"
  password: "${REDIS_PASSWORD:}"
  database: 0
  pool_size: 20

messaging:
  type: "redis_streams"  # redis_streams, kafka, rabbitmq
  config:
    stream_prefix: "funding_arb"
    consumer_group: "main_group"
    block_timeout: 1000
    max_retries: 3

api:
  rest:
    host: "0.0.0.0"
    port: 8000
    cors_origins: ["*"]
    rate_limit: 1000

  websocket:
    host: "0.0.0.0"
    port: 8001
    max_connections: 1000
    ping_interval: 30

monitoring:
  enabled: true
  prometheus:
    enabled: true
    port: 9090
    path: "/metrics"

  grafana:
    enabled: true
    port: 3000

  alerting:
    enabled: true
    webhook_url: "${ALERT_WEBHOOK_URL:}"
    email:
      smtp_host: "${SMTP_HOST:}"
      smtp_port: "${SMTP_PORT:587}"
      username: "${SMTP_USER:}"
      password: "${SMTP_PASSWORD:}"
      from_address: "${ALERT_FROM_EMAIL:}"
      to_addresses:
        - "${ALERT_TO_EMAIL:}"
```

### 2.2 交易所配置

```yaml
# config/exchanges.yaml
exchanges:
  binance:
    enabled: true
    plugin_type: "ccxt"
    api_key: "${BINANCE_API_KEY:}"
    secret_key: "${BINANCE_SECRET_KEY:}"
    sandbox: false
    rate_limit: 1200
    timeout: 30000
    symbols:
      - "BTC/USDT"
      - "ETH/USDT"
      - "BNB/USDT"
    features:
      spot_trading: true
      futures_trading: true
      funding_rate: true

  okx:
    enabled: true
    plugin_type: "ccxt"
    api_key: "${OKX_API_KEY:}"
    secret_key: "${OKX_SECRET_KEY:}"
    passphrase: "${OKX_PASSPHRASE:}"
    sandbox: false
    rate_limit: 1000
    timeout: 30000
    symbols:
      - "BTC/USDT"
      - "ETH/USDT"
      - "OKB/USDT"
    features:
      spot_trading: true
      futures_trading: true
      funding_rate: true

  bybit:
    enabled: true
    plugin_type: "custom"
    api_key: "${BYBIT_API_KEY:}"
    secret_key: "${BYBIT_SECRET_KEY:}"
    base_url: "https://api.bybit.com"
    sandbox: false
    rate_limit: 600
    timeout: 30000
    symbols:
      - "BTC/USDT"
      - "ETH/USDT"
    features:
      spot_trading: true
      futures_trading: true
      funding_rate: true
```

### 2.3 策略配置

```yaml
# config/strategies.yaml
strategies:
  funding_rate_arbitrage:
    enabled: true
    class: "FundingRateArbitrageStrategy"
    priority: 1
    symbols:
      - "BTC/USDT"
      - "ETH/USDT"
    exchanges:
      - "binance"
      - "okx"
      - "bybit"
    parameters:
      min_spread: 0.0001  # 最小价差 0.01%
      max_spread: 0.01    # 最大价差 1%
      position_size: 1000  # 每次交易金额 (USDT)
      max_positions: 5     # 最大同时持仓数
      stop_loss: 0.005     # 止损比例 0.5%
      take_profit: 0.003   # 止盈比例 0.3%
      funding_threshold: 0.0005  # 资金费率阈值 0.05%

  basis_arbitrage:
    enabled: false
    class: "BasisArbitrageStrategy"
    priority: 2
    symbols:
      - "BTC/USDT"
    parameters:
      min_basis: 0.002
      max_basis: 0.02
      position_size: 500
      max_positions: 3
```

### 2.4 风险管理配置

```yaml
# config/risk.yaml
risk_management:
  global_limits:
    max_daily_loss: 5000      # 最大日损失 (USDT)
    max_drawdown: 0.1         # 最大回撤 10%
    max_position_size: 10000  # 最大单笔仓位 (USDT)
    max_leverage: 5           # 最大杠杆倍数

  position_limits:
    BTC/USDT:
      max_position: 2.0       # 最大持仓量 (BTC)
      max_concentration: 0.3  # 最大集中度 30%
    ETH/USDT:
      max_position: 50.0      # 最大持仓量 (ETH)
      max_concentration: 0.2  # 最大集中度 20%

  var_limits:
    confidence_level: 0.95
    time_horizon: 1          # 1天
    max_var: 1000           # 最大VaR (USDT)

  risk_alerts:
    var_breach:
      threshold: 0.8          # VaR使用率80%告警
      action: "warn"
    position_limit_breach:
      threshold: 0.9          # 仓位限制90%告警
      action: "reduce_position"
    daily_loss_breach:
      threshold: 0.8          # 日损失80%告警
      action: "emergency_stop"
```

## 3. 配置管理实现

### 3.1 配置加载器

```python
import yaml
import json
import os
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    host: str
    port: int
    database: str
    username: str
    password: str
    pool_size: int = 20
    max_overflow: int = 30

@dataclass
class RedisConfig:
    host: str
    port: int
    password: str = ""
    database: int = 0
    pool_size: int = 20

@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "json"
    output: str = "stdout"
    rotation: Dict[str, Union[str, int]] = None

@dataclass
class ExchangeConfig:
    name: str
    enabled: bool
    plugin_type: str
    api_key: str
    secret_key: str
    passphrase: str = ""
    sandbox: bool = False
    rate_limit: int = 1000
    timeout: int = 30000
    symbols: List[str] = None
    features: Dict[str, bool] = None

@dataclass
class StrategyConfig:
    name: str
    enabled: bool
    class_name: str
    priority: int
    symbols: List[str]
    exchanges: List[str]
    parameters: Dict[str, Any]

@dataclass
class RiskConfig:
    global_limits: Dict[str, Union[int, float]]
    position_limits: Dict[str, Dict[str, float]]
    var_limits: Dict[str, Union[float, int]]
    risk_alerts: Dict[str, Dict[str, Union[str, float]]]

@dataclass
class SystemConfig:
    system: Dict[str, Any]
    logging: LoggingConfig
    database: Dict[str, DatabaseConfig]
    redis: RedisConfig
    messaging: Dict[str, Any]
    api: Dict[str, Any]
    monitoring: Dict[str, Any]
    exchanges: List[ExchangeConfig]
    strategies: List[StrategyConfig]
    risk: RiskConfig

class ConfigLoader:
    """配置加载器"""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.env_vars_cache: Dict[str, str] = {}

    def load_config(self) -> SystemConfig:
        """加载完整配置"""
        # 加载主配置
        main_config = self._load_yaml_file("main.yaml")

        # 加载交易所配置
        exchanges_config = self._load_yaml_file("exchanges.yaml")

        # 加载策略配置
        strategies_config = self._load_yaml_file("strategies.yaml")

        # 加载风险配置
        risk_config = self._load_yaml_file("risk.yaml")

        # 合并配置
        full_config = {
            **main_config,
            **exchanges_config,
            **strategies_config,
            **risk_config
        }

        # 环境变量替换
        full_config = self._resolve_env_vars(full_config)

        # 验证配置
        self._validate_config(full_config)

        return self._parse_config(full_config)

    def _load_yaml_file(self, filename: str) -> Dict[str, Any]:
        """加载YAML文件"""
        file_path = self.config_dir / filename

        if not file_path.exists():
            logger.warning(f"Config file not found: {file_path}")
            return {}

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to load config file {filename}: {e}")
            raise

    def _resolve_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """解析环境变量"""
        def resolve_value(value):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # 解析 ${VAR_NAME:default_value} 格式
                var_spec = value[2:-1]
                if ":" in var_spec:
                    var_name, default_value = var_spec.split(":", 1)
                else:
                    var_name, default_value = var_spec, ""

                return os.getenv(var_name, default_value)

            elif isinstance(value, dict):
                return {k: resolve_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [resolve_value(item) for item in value]
            else:
                return value

        return resolve_value(config)

    def _validate_config(self, config: Dict[str, Any]):
        """验证配置"""
        required_sections = ['system', 'database', 'redis']

        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required config section: {section}")

        # 验证数据库配置
        db_config = config.get('database', {})
        if 'postgresql' in db_config:
            pg_config = db_config['postgresql']
            required_pg_fields = ['host', 'port', 'database', 'username', 'password']

            for field in required_pg_fields:
                if not pg_config.get(field):
                    raise ValueError(f"Missing PostgreSQL config field: {field}")

    def _parse_config(self, config: Dict[str, Any]) -> SystemConfig:
        """解析配置为数据类"""
        # 解析日志配置
        logging_config = LoggingConfig(**config.get('logging', {}))

        # 解析数据库配置
        database_configs = {}
        for db_name, db_config in config.get('database', {}).items():
            database_configs[db_name] = DatabaseConfig(**db_config)

        # 解析Redis配置
        redis_config = RedisConfig(**config.get('redis', {}))

        # 解析交易所配置
        exchanges = []
        for exchange_name, exchange_config in config.get('exchanges', {}).items():
            exchanges.append(ExchangeConfig(
                name=exchange_name,
                **exchange_config
            ))

        # 解析策略配置
        strategies = []
        for strategy_name, strategy_config in config.get('strategies', {}).items():
            strategies.append(StrategyConfig(
                name=strategy_name,
                class_name=strategy_config.get('class', strategy_name),
                **{k: v for k, v in strategy_config.items() if k != 'class'}
            ))

        # 解析风险配置
        risk_config = RiskConfig(**config.get('risk_management', {}))

        return SystemConfig(
            system=config.get('system', {}),
            logging=logging_config,
            database=database_configs,
            redis=redis_config,
            messaging=config.get('messaging', {}),
            api=config.get('api', {}),
            monitoring=config.get('monitoring', {}),
            exchanges=exchanges,
            strategies=strategies,
            risk=risk_config
        )

class ConfigManager:
    """配置管理器"""

    def __init__(self, config_dir: str = "config", watch_changes: bool = True):
        self.config_dir = config_dir
        self.watch_changes = watch_changes
        self.loader = ConfigLoader(config_dir)
        self.current_config: Optional[SystemConfig] = None
        self.change_handlers: List[Callable] = []

    async def initialize(self):
        """初始化配置管理器"""
        self.current_config = self.loader.load_config()

        if self.watch_changes:
            # 启动配置文件监控
            asyncio.create_task(self._watch_config_changes())

    def get_config(self) -> SystemConfig:
        """获取当前配置"""
        return self.current_config

    def get_exchange_config(self, name: str) -> Optional[ExchangeConfig]:
        """获取交易所配置"""
        for exchange in self.current_config.exchanges:
            if exchange.name == name:
                return exchange
        return None

    def get_strategy_config(self, name: str) -> Optional[StrategyConfig]:
        """获取策略配置"""
        for strategy in self.current_config.strategies:
            if strategy.name == name:
                return strategy
        return None

    def register_change_handler(self, handler: Callable):
        """注册配置变更处理器"""
        self.change_handlers.append(handler)

    async def reload_config(self):
        """重新加载配置"""
        try:
            new_config = self.loader.load_config()
            old_config = self.current_config
            self.current_config = new_config

            # 通知变更处理器
            for handler in self.change_handlers:
                try:
                    await handler(old_config, new_config)
                except Exception as e:
                    logger.error(f"Config change handler error: {e}")

            logger.info("Configuration reloaded successfully")

        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
            raise

    async def _watch_config_changes(self):
        """监控配置文件变更"""
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        class ConfigChangeHandler(FileSystemEventHandler):
            def __init__(self, config_manager):
                self.config_manager = config_manager

            def on_modified(self, event):
                if event.is_directory:
                    return

                if event.src_path.endswith(('.yaml', '.yml', '.json')):
                    logger.info(f"Config file changed: {event.src_path}")
                    asyncio.create_task(self.config_manager.reload_config())

        observer = Observer()
        observer.schedule(
            ConfigChangeHandler(self),
            self.config_dir,
            recursive=True
        )
        observer.start()

        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            observer.stop()
            observer.join()

class ConfigValidator:
    """配置验证器"""

    @staticmethod
    def validate_exchange_config(config: ExchangeConfig) -> List[str]:
        """验证交易所配置"""
        errors = []

        if not config.api_key:
            errors.append(f"Missing API key for exchange {config.name}")

        if not config.secret_key:
            errors.append(f"Missing secret key for exchange {config.name}")

        if config.rate_limit <= 0:
            errors.append(f"Invalid rate limit for exchange {config.name}")

        return errors

    @staticmethod
    def validate_strategy_config(config: StrategyConfig) -> List[str]:
        """验证策略配置"""
        errors = []

        if not config.symbols:
            errors.append(f"No symbols configured for strategy {config.name}")

        if not config.exchanges:
            errors.append(f"No exchanges configured for strategy {config.name}")

        if not config.parameters:
            errors.append(f"No parameters configured for strategy {config.name}")

        return errors

    @staticmethod
    def validate_risk_config(config: RiskConfig) -> List[str]:
        """验证风险配置"""
        errors = []

        if not config.global_limits:
            errors.append("Missing global risk limits")

        required_limits = ['max_daily_loss', 'max_drawdown', 'max_position_size']
        for limit in required_limits:
            if limit not in config.global_limits:
                errors.append(f"Missing risk limit: {limit}")

        return errors
```

## 4. 容器化部署

### 4.1 Docker配置

```dockerfile
# Dockerfile
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建非root用户
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 暴露端口
EXPOSE 8000 8001 9090

# 启动命令
CMD ["python", "main.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    container_name: funding-rate-arb
    restart: unless-stopped
    ports:
      - "8000:8000"  # REST API
      - "8001:8001"  # WebSocket
      - "9090:9090"  # Prometheus metrics
    environment:
      - ENVIRONMENT=production
      - DB_HOST=postgres
      - DB_PASSWORD=${DB_PASSWORD}
      - REDIS_HOST=redis
      - INFLUX_HOST=influxdb
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
      - ./data:/app/data
    depends_on:
      - postgres
      - redis
      - influxdb
    networks:
      - funding-arb-network

  postgres:
    image: postgres:13
    container_name: funding-arb-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_DB=funding_arb
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=${DB_PASSWORD:-postgres}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./database/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    networks:
      - funding-arb-network

  redis:
    image: redis:6-alpine
    container_name: funding-arb-redis
    restart: unless-stopped
    command: redis-server --requirepass ${REDIS_PASSWORD:-redis123}
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    networks:
      - funding-arb-network

  influxdb:
    image: influxdb:1.8
    container_name: funding-arb-influxdb
    restart: unless-stopped
    environment:
      - INFLUXDB_DB=market_data
      - INFLUXDB_ADMIN_USER=admin
      - INFLUXDB_ADMIN_PASSWORD=${INFLUX_PASSWORD:-admin123}
    volumes:
      - influxdb_data:/var/lib/influxdb
    ports:
      - "8086:8086"
    networks:
      - funding-arb-network

  prometheus:
    image: prom/prometheus:latest
    container_name: funding-arb-prometheus
    restart: unless-stopped
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
    ports:
      - "9091:9090"
    networks:
      - funding-arb-network

  grafana:
    image: grafana/grafana:latest
    container_name: funding-arb-grafana
    restart: unless-stopped
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin123}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
    ports:
      - "3000:3000"
    networks:
      - funding-arb-network

networks:
  funding-arb-network:
    driver: bridge

volumes:
  postgres_data:
  redis_data:
  influxdb_data:
  prometheus_data:
  grafana_data:
```

### 4.2 Kubernetes部署

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: funding-arb
---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: funding-arb-config
  namespace: funding-arb
data:
  main.yaml: |
    system:
      name: "funding-rate-arb"
      version: "1.0.0"
      environment: "production"
    logging:
      level: "INFO"
      format: "json"
    database:
      postgresql:
        host: "postgres-service"
        port: "5432"
        database: "funding_arb"
        username: "postgres"
        password: "${DB_PASSWORD}"
    redis:
      host: "redis-service"
      port: "6379"
---
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: funding-arb-secrets
  namespace: funding-arb
type: Opaque
data:
  db-password: cG9zdGdyZXM=  # base64 encoded
  binance-api-key: ""
  binance-secret-key: ""
  okx-api-key: ""
  okx-secret-key: ""
  okx-passphrase: ""
---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: funding-arb-app
  namespace: funding-arb
  labels:
    app: funding-arb
spec:
  replicas: 2
  selector:
    matchLabels:
      app: funding-arb
  template:
    metadata:
      labels:
        app: funding-arb
    spec:
      containers:
      - name: funding-arb
        image: funding-arb:latest
        ports:
        - containerPort: 8000
        - containerPort: 8001
        - containerPort: 9090
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DB_PASSWORD
          valueFrom:
            secretKeyRef:
              name: funding-arb-secrets
              key: db-password
        - name: BINANCE_API_KEY
          valueFrom:
            secretKeyRef:
              name: funding-arb-secrets
              key: binance-api-key
        volumeMounts:
        - name: config-volume
          mountPath: /app/config
          readOnly: true
        - name: logs-volume
          mountPath: /app/logs
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config-volume
        configMap:
          name: funding-arb-config
      - name: logs-volume
        emptyDir: {}
---
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: funding-arb-service
  namespace: funding-arb
spec:
  selector:
    app: funding-arb
  ports:
  - name: rest-api
    port: 8000
    targetPort: 8000
  - name: websocket
    port: 8001
    targetPort: 8001
  - name: metrics
    port: 9090
    targetPort: 9090
  type: LoadBalancer
```

### 4.3 部署脚本

```bash
#!/bin/bash
# deploy.sh

set -e

ENVIRONMENT=${1:-development}
NAMESPACE="funding-arb"

echo "Deploying funding-rate-arb to $ENVIRONMENT environment..."

# 检查kubectl连接
if ! kubectl cluster-info > /dev/null 2>&1; then
    echo "Error: Unable to connect to Kubernetes cluster"
    exit 1
fi

# 创建命名空间
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# 部署配置
echo "Deploying configuration..."
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secret.yaml

# 部署数据库
echo "Deploying database..."
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/influxdb.yaml

# 等待数据库就绪
echo "Waiting for database to be ready..."
kubectl wait --for=condition=ready pod -l app=postgres -n $NAMESPACE --timeout=300s
kubectl wait --for=condition=ready pod -l app=redis -n $NAMESPACE --timeout=300s

# 部署应用
echo "Deploying application..."
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# 部署监控
echo "Deploying monitoring..."
kubectl apply -f k8s/prometheus.yaml
kubectl apply -f k8s/grafana.yaml

# 等待应用就绪
echo "Waiting for application to be ready..."
kubectl wait --for=condition=ready pod -l app=funding-arb -n $NAMESPACE --timeout=300s

# 显示部署状态
echo "Deployment completed successfully!"
echo ""
echo "Services:"
kubectl get services -n $NAMESPACE

echo ""
echo "Pods:"
kubectl get pods -n $NAMESPACE

echo ""
echo "Application URL:"
SERVICE_IP=$(kubectl get service funding-arb-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
if [ -n "$SERVICE_IP" ]; then
    echo "REST API: http://$SERVICE_IP:8000"
    echo "WebSocket: ws://$SERVICE_IP:8001"
    echo "Metrics: http://$SERVICE_IP:9090/metrics"
else
    echo "Service is not yet accessible externally. Use port-forward:"
    echo "kubectl port-forward -n $NAMESPACE service/funding-arb-service 8000:8000"
fi
```

## 5. 监控与告警配置

### 5.1 Prometheus配置

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'funding-arb'
    static_configs:
      - targets: ['app:9090']
    scrape_interval: 5s
    metrics_path: /metrics

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:9121']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### 5.2 告警规则

```yaml
# monitoring/rules/alerts.yml
groups:
- name: funding-arb
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is above 10% for more than 2 minutes"

  - alert: HighLatency
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 0.5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High latency detected"
      description: "95th percentile latency is above 500ms for more than 5 minutes"

  - alert: DatabaseConnectionFailed
    expr: up{job="postgres"} == 0
    for: 30s
    labels:
      severity: critical
    annotations:
      summary: "Database connection failed"
      description: "Unable to connect to PostgreSQL database"

  - alert: RedisConnectionFailed
    expr: up{job="redis"} == 0
    for: 30s
    labels:
      severity: critical
    annotations:
      summary: "Redis connection failed"
      description: "Unable to connect to Redis"

  - alert: ExchangeConnectionLost
    expr: exchange_connection_status != 1
    for: 1m
    labels:
      severity: warning
    annotations:
      summary: "Exchange connection lost"
      description: "Connection to exchange {{ $labels.exchange }} is down"

  - alert: RiskLimitBreach
    expr: risk_var_utilization > 0.9
    for: 0s
    labels:
      severity: critical
    annotations:
      summary: "Risk limit breach"
      description: "VaR utilization is above 90%"

  - alert: LowProfitability
    expr: strategy_daily_pnl < -1000
    for: 30m
    labels:
      severity: warning
    annotations:
      summary: "Low profitability"
      description: "Daily P&L is below -$1000 for 30 minutes"
```

### 5.3 Grafana仪表板

```json
{
  "dashboard": {
    "id": null,
    "title": "Funding Rate Arbitrage Dashboard",
    "tags": ["funding", "arbitrage"],
    "timezone": "UTC",
    "panels": [
      {
        "id": 1,
        "title": "System Overview",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"funding-arb\"}",
            "legendFormat": "System Status"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "color": {
              "mode": "thresholds"
            },
            "thresholds": {
              "steps": [
                {"color": "red", "value": 0},
                {"color": "green", "value": 1}
              ]
            }
          }
        }
      },
      {
        "id": 2,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ],
        "yAxes": [
          {
            "label": "Requests/sec"
          }
        ]
      },
      {
        "id": 3,
        "title": "Exchange Connections",
        "type": "stat",
        "targets": [
          {
            "expr": "exchange_connection_status",
            "legendFormat": "{{exchange}}"
          }
        ]
      },
      {
        "id": 4,
        "title": "Portfolio P&L",
        "type": "graph",
        "targets": [
          {
            "expr": "portfolio_total_pnl",
            "legendFormat": "Total P&L"
          }
        ],
        "yAxes": [
          {
            "label": "P&L (USD)"
          }
        ]
      },
      {
        "id": 5,
        "title": "Risk Metrics",
        "type": "graph",
        "targets": [
          {
            "expr": "risk_var_current",
            "legendFormat": "Current VaR"
          },
          {
            "expr": "risk_var_limit",
            "legendFormat": "VaR Limit"
          }
        ]
      },
      {
        "id": 6,
        "title": "Active Positions",
        "type": "table",
        "targets": [
          {
            "expr": "position_size",
            "format": "table"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

## 6. 运维脚本

### 6.1 健康检查脚本

```bash
#!/bin/bash
# scripts/health_check.sh

NAMESPACE="funding-arb"
SERVICE_NAME="funding-arb-service"

echo "=== Funding Rate Arbitrage Health Check ==="

# 检查Pod状态
echo "1. Checking pod status..."
kubectl get pods -n $NAMESPACE -l app=funding-arb

# 检查服务状态
echo "2. Checking service status..."
kubectl get services -n $NAMESPACE

# 检查应用健康端点
echo "3. Checking application health..."
SERVICE_IP=$(kubectl get service $SERVICE_NAME -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

if [ -n "$SERVICE_IP" ]; then
    curl -f http://$SERVICE_IP:8000/health || echo "Health check failed"
    curl -f http://$SERVICE_IP:8000/metrics || echo "Metrics endpoint failed"
else
    echo "Service IP not available, using port-forward..."
    kubectl port-forward -n $NAMESPACE service/$SERVICE_NAME 8000:8000 &
    PF_PID=$!
    sleep 2

    curl -f http://localhost:8000/health || echo "Health check failed"
    curl -f http://localhost:8000/metrics || echo "Metrics endpoint failed"

    kill $PF_PID
fi

# 检查数据库连接
echo "4. Checking database connectivity..."
kubectl exec -n $NAMESPACE deployment/funding-arb-app -- python -c "
import psycopg2
try:
    conn = psycopg2.connect(host='postgres-service', database='funding_arb', user='postgres', password='postgres')
    print('PostgreSQL: OK')
    conn.close()
except Exception as e:
    print(f'PostgreSQL: ERROR - {e}')
"

# 检查Redis连接
echo "5. Checking Redis connectivity..."
kubectl exec -n $NAMESPACE deployment/funding-arb-app -- python -c "
import redis
try:
    r = redis.Redis(host='redis-service', port=6379, decode_responses=True)
    r.ping()
    print('Redis: OK')
except Exception as e:
    print(f'Redis: ERROR - {e}')
"

echo "=== Health Check Complete ==="
```

### 6.2 备份脚本

```bash
#!/bin/bash
# scripts/backup.sh

NAMESPACE="funding-arb"
BACKUP_DIR="/backup/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

echo "Starting backup to $BACKUP_DIR..."

# 备份PostgreSQL数据库
echo "Backing up PostgreSQL..."
kubectl exec -n $NAMESPACE deployment/postgres -- pg_dump -U postgres funding_arb > $BACKUP_DIR/postgres_backup.sql

# 备份Redis数据
echo "Backing up Redis..."
kubectl exec -n $NAMESPACE deployment/redis -- redis-cli BGSAVE
kubectl cp $NAMESPACE/redis:/data/dump.rdb $BACKUP_DIR/redis_backup.rdb

# 备份配置文件
echo "Backing up configuration..."
kubectl get configmap -n $NAMESPACE -o yaml > $BACKUP_DIR/configmaps.yaml
kubectl get secret -n $NAMESPACE -o yaml > $BACKUP_DIR/secrets.yaml

# 备份InfluxDB数据
echo "Backing up InfluxDB..."
kubectl exec -n $NAMESPACE deployment/influxdb -- influxd backup -portable /backup
kubectl cp $NAMESPACE/influxdb:/backup $BACKUP_DIR/influxdb_backup

# 压缩备份
echo "Compressing backup..."
tar -czf $BACKUP_DIR.tar.gz -C $(dirname $BACKUP_DIR) $(basename $BACKUP_DIR)
rm -rf $BACKUP_DIR

echo "Backup completed: $BACKUP_DIR.tar.gz"
```

### 6.3 升级脚本

```bash
#!/bin/bash
# scripts/upgrade.sh

set -e

NAMESPACE="funding-arb"
IMAGE_TAG=${1:-latest}

echo "Upgrading funding-rate-arb to version $IMAGE_TAG..."

# 备份当前版本
echo "Creating backup..."
./scripts/backup.sh

# 更新镜像
echo "Updating container image..."
kubectl set image deployment/funding-arb-app -n $NAMESPACE funding-arb=funding-arb:$IMAGE_TAG

# 等待滚动更新完成
echo "Waiting for rollout to complete..."
kubectl rollout status deployment/funding-arb-app -n $NAMESPACE

# 验证升级
echo "Verifying upgrade..."
kubectl get pods -n $NAMESPACE -l app=funding-arb

# 运行健康检查
echo "Running health check..."
./scripts/health_check.sh

echo "Upgrade completed successfully!"
```

这个配置管理与部署方案提供了完整的系统配置框架和部署解决方案，支持多环境管理、容器化部署、Kubernetes编排、监控告警等功能，确保系统能够稳定、可靠地运行在生产环境中。