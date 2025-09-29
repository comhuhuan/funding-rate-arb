#!/usr/bin/env python3
"""
Main entry point for the Funding Rate Arbitrage System.

This module initializes and starts the entire trading system with all its components.
"""

import asyncio
import signal
import sys
import logging
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.infrastructure.logging import setup_logging
from src.infrastructure.config import ConfigManager
from src.system_manager import SystemManager

logger = logging.getLogger(__name__)


async def main():
    """Main application entry point."""
    system_manager = None

    try:
        # Setup logging
        setup_logging()
        logger.info("Starting Funding Rate Arbitrage System...")

        # Load configuration
        config_manager = ConfigManager("config")
        await config_manager.initialize()
        config = config_manager.get_config()

        logger.info(f"System: {config.system['name']} v{config.system['version']}")
        logger.info(f"Environment: {config.system['environment']}")

        # Initialize system manager
        system_manager = SystemManager(config_manager)

        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            if system_manager:
                asyncio.create_task(system_manager.shutdown())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start the system
        await system_manager.start()

        # Keep the system running
        logger.info("System started successfully. Press Ctrl+C to stop.")

        # Wait for shutdown signal
        try:
            while system_manager.is_running:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")

    except Exception as e:
        logger.error(f"System startup failed: {e}", exc_info=True)
        return 1
    finally:
        if system_manager:
            await system_manager.shutdown()
        logger.info("System shutdown complete")

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))