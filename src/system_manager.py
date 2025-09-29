"""
System Manager for the Funding Rate Arbitrage System.

This module provides the main SystemManager class that orchestrates
the initialization, startup, and shutdown of all system components.
"""

import asyncio
import logging
from typing import Optional

from src.infrastructure.config import ConfigManager

logger = logging.getLogger(__name__)


class SystemManager:
    """
    Main system manager that orchestrates all components.

    This class is responsible for:
    - Initializing all system components
    - Managing the startup and shutdown sequence
    - Coordinating between different modules
    - Handling system-wide error conditions
    """

    def __init__(self, config_manager: ConfigManager):
        """
        Initialize the system manager.

        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
        self.is_running = False
        self._shutdown_event = asyncio.Event()

        # Component managers (to be initialized)
        self.exchange_manager: Optional[object] = None
        self.market_data_service: Optional[object] = None
        self.funding_rate_service: Optional[object] = None
        self.arbitrage_engine: Optional[object] = None
        self.strategy_manager: Optional[object] = None
        self.order_manager: Optional[object] = None
        self.risk_manager: Optional[object] = None
        self.api_server: Optional[object] = None

    async def start(self) -> None:
        """
        Start the entire trading system.

        This method initializes and starts all components in the correct order.
        """
        try:
            logger.info("Starting Funding Rate Arbitrage System...")

            # TODO: Initialize components in dependency order
            # 1. Infrastructure layer (cache, messaging)
            # 2. Exchange layer
            # 3. Data layer (market data, funding rates)
            # 4. Core layer (arbitrage, risk, strategies)
            # 5. Trading layer (order management)
            # 6. API layer

            self.is_running = True
            logger.info("System startup completed successfully")

            # Wait for shutdown signal
            await self._shutdown_event.wait()

        except Exception as e:
            logger.error(f"System startup failed: {e}", exc_info=True)
            self.is_running = False
            raise

    async def shutdown(self) -> None:
        """
        Gracefully shutdown the trading system.

        This method stops all components in reverse order to ensure
        clean shutdown and data integrity.
        """
        logger.info("Initiating system shutdown...")

        try:
            # Set shutdown flag
            self.is_running = False
            self._shutdown_event.set()

            # TODO: Shutdown components in reverse order
            # 1. API layer
            # 2. Trading layer
            # 3. Core layer
            # 4. Data layer
            # 5. Exchange layer
            # 6. Infrastructure layer

            logger.info("System shutdown completed successfully")

        except Exception as e:
            logger.error(f"Error during system shutdown: {e}", exc_info=True)

    async def get_system_status(self) -> dict:
        """
        Get current system status.

        Returns:
            Dictionary containing status of all components
        """
        return {
            "system_running": self.is_running,
            "components": {
                "exchange_manager": self.exchange_manager is not None,
                "market_data_service": self.market_data_service is not None,
                "funding_rate_service": self.funding_rate_service is not None,
                "arbitrage_engine": self.arbitrage_engine is not None,
                "strategy_manager": self.strategy_manager is not None,
                "order_manager": self.order_manager is not None,
                "risk_manager": self.risk_manager is not None,
                "api_server": self.api_server is not None,
            }
        }