#!/usr/bin/env python3
"""Real-time WebSocket Streaming Service with WSS (WebSocket Secure) Support
Provides secure real-time data streaming for MFC monitoring dashboard.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import signal
import ssl
import sys
import threading
from datetime import datetime
from pathlib import Path

import pandas as pd
import websockets

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import contextlib
from typing import TYPE_CHECKING

from monitoring.ssl_config import (
    SSLConfig,
    SSLContextManager,
    initialize_ssl_infrastructure,
    load_ssl_config,
    test_ssl_connection,
)

if TYPE_CHECKING:
    from websockets.server import WebSocketServerProtocol

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class DataStreamManager:
    """Manages real-time data streaming with secure WebSocket connections."""

    def __init__(self, ssl_config: SSLConfig | None = None) -> None:
        self.ssl_config = ssl_config or load_ssl_config()
        self.clients: set[WebSocketServerProtocol] = set()
        self.data_cache: list[dict] = []
        self.cache_size_limit = 1000
        self.last_data_time = datetime.now()
        self.streaming_active = False
        self.data_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # SSL context manager
        self.ssl_context_manager = None
        if self.ssl_config:
            self.ssl_context_manager = SSLContextManager(self.ssl_config)

    async def register_client(self, websocket: WebSocketServerProtocol) -> None:
        """Register new WebSocket client."""
        self.clients.add(websocket)
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"Client connected: {client_info} (Total: {len(self.clients)})")

        # Send initial data if available
        if self.data_cache:
            await self.send_to_client(
                websocket,
                {
                    "type": "initial_data",
                    "data": self.data_cache[-100:],  # Last 100 points
                    "timestamp": datetime.now().isoformat(),
                },
            )

    async def unregister_client(self, websocket: WebSocketServerProtocol) -> None:
        """Unregister WebSocket client."""
        self.clients.discard(websocket)
        client_info = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"Client disconnected: {client_info} (Total: {len(self.clients)})")

    async def send_to_client(
        self,
        websocket: WebSocketServerProtocol,
        message: dict,
    ) -> None:
        """Send message to specific client with error handling."""
        try:
            await websocket.send(json.dumps(message))
        except websockets.exceptions.ConnectionClosed:
            await self.unregister_client(websocket)
        except Exception as e:
            logger.exception(f"Error sending to client: {e}")
            await self.unregister_client(websocket)

    async def broadcast_to_clients(self, message: dict) -> None:
        """Broadcast message to all connected clients."""
        if not self.clients:
            return

        # Create list copy to avoid modification during iteration
        clients_copy = list(self.clients)

        # Send to all clients concurrently
        tasks = [self.send_to_client(client, message) for client in clients_copy]
        await asyncio.gather(*tasks, return_exceptions=True)

    def load_simulation_data(self) -> list[dict] | None:
        """Load latest simulation data from files."""
        try:
            # Find latest simulation data directory
            data_dir = Path("../../../data/simulation_data")

            if not data_dir.exists():
                return None

            # Find most recent simulation
            simulation_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
            if not simulation_dirs:
                return None

            latest_dir = max(simulation_dirs, key=lambda d: d.stat().st_mtime)

            # Find compressed CSV data file
            csv_files = list(latest_dir.glob("*_data_*.csv.gz"))
            if not csv_files:
                return None

            latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)

            # Check if file has been modified since last load
            file_mtime = datetime.fromtimestamp(latest_file.stat().st_mtime)
            if file_mtime <= self.last_data_time:
                return None  # No new data

            self.last_data_time = file_mtime

            # Load data
            with gzip.open(latest_file, "rt") as f:
                df = pd.read_csv(f)

            # Convert to list of dictionaries
            data_points = []
            for _, row in df.iterrows():
                # Parse biofilm thickness data safely
                try:
                    if isinstance(row["biofilm_thicknesses"], str):
                        biofilm_thicknesses = json.loads(
                            row["biofilm_thicknesses"].replace("'", '"'),
                        )
                    else:
                        biofilm_thicknesses = [float(row["biofilm_thicknesses"])]
                except Exception:
                    biofilm_thicknesses = [10.0]  # Default value

                data_point = {
                    "timestamp": datetime.now().isoformat(),
                    "time_hours": float(row["time_hours"]),
                    "reservoir_concentration": float(row["reservoir_concentration"]),
                    "outlet_concentration": float(row["outlet_concentration"]),
                    "total_power": float(row["total_power"]),
                    "biofilm_thicknesses": biofilm_thicknesses,
                    "substrate_addition_rate": float(row["substrate_addition_rate"]),
                    "q_action": int(row["q_action"]),
                    "epsilon": float(row["epsilon"]),
                    "reward": float(row["reward"]),
                }
                data_points.append(data_point)

            return data_points

        except Exception as e:
            logger.exception(f"Error loading simulation data: {e}")
            return None

    def update_data_cache(self, new_data: list[dict]) -> None:
        """Update internal data cache with size limit."""
        if new_data:
            self.data_cache.extend(new_data)

            # Maintain cache size limit
            if len(self.data_cache) > self.cache_size_limit:
                self.data_cache = self.data_cache[-self.cache_size_limit :]

    async def data_polling_loop(self) -> None:
        """Continuous data polling and broadcasting loop."""
        logger.info("Starting data polling loop")

        while not self._stop_event.is_set():
            try:
                # Load new simulation data
                new_data = self.load_simulation_data()

                if new_data:
                    # Update cache
                    self.update_data_cache(new_data)

                    # Broadcast to connected clients
                    message = {
                        "type": "data_update",
                        "data": new_data[-10:],  # Send last 10 points
                        "total_points": len(self.data_cache),
                        "timestamp": datetime.now().isoformat(),
                    }

                    await self.broadcast_to_clients(message)
                    logger.debug(
                        f"Broadcasted {len(new_data)} new data points to {len(self.clients)} clients",
                    )

                # Send heartbeat to maintain connections
                elif self.clients:
                    heartbeat = {
                        "type": "heartbeat",
                        "timestamp": datetime.now().isoformat(),
                        "clients_connected": len(self.clients),
                    }

                    await self.broadcast_to_clients(heartbeat)

                # Wait before next poll
                await asyncio.sleep(2.0)  # Poll every 2 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Error in data polling loop: {e}")
                await asyncio.sleep(5.0)  # Wait longer on error

        logger.info("Data polling loop stopped")

    async def handle_client_message(
        self,
        websocket: WebSocketServerProtocol,
        message: str,
    ) -> None:
        """Handle incoming messages from clients."""
        try:
            data = json.loads(message)
            message_type = data.get("type")

            if message_type == "ping":
                # Respond to ping with pong
                await self.send_to_client(
                    websocket,
                    {"type": "pong", "timestamp": datetime.now().isoformat()},
                )

            elif message_type == "request_data":
                # Send recent data
                limit = data.get("limit", 100)
                recent_data = self.data_cache[-limit:]

                await self.send_to_client(
                    websocket,
                    {
                        "type": "data_response",
                        "data": recent_data,
                        "timestamp": datetime.now().isoformat(),
                    },
                )

            elif message_type == "subscribe":
                # Subscribe to specific data streams (future feature)
                streams = data.get("streams", ["all"])
                await self.send_to_client(
                    websocket,
                    {
                        "type": "subscription_confirmed",
                        "streams": streams,
                        "timestamp": datetime.now().isoformat(),
                    },
                )

            else:
                logger.warning(f"Unknown message type: {message_type}")
                await self.send_to_client(
                    websocket,
                    {
                        "type": "error",
                        "message": f"Unknown message type: {message_type}",
                        "timestamp": datetime.now().isoformat(),
                    },
                )

        except json.JSONDecodeError:
            await self.send_to_client(
                websocket,
                {
                    "type": "error",
                    "message": "Invalid JSON message",
                    "timestamp": datetime.now().isoformat(),
                },
            )
        except Exception as e:
            logger.exception(f"Error handling client message: {e}")
            await self.send_to_client(
                websocket,
                {
                    "type": "error",
                    "message": "Internal server error",
                    "timestamp": datetime.now().isoformat(),
                },
            )

    async def client_handler(
        self,
        websocket: WebSocketServerProtocol,
        path: str,
    ) -> None:
        """Handle individual WebSocket client connections."""
        await self.register_client(websocket)

        try:
            async for message in websocket:
                await self.handle_client_message(websocket, message)

        except websockets.exceptions.ConnectionClosed:
            pass  # Normal disconnection
        except Exception as e:
            logger.exception(f"Error in client handler: {e}")
        finally:
            await self.unregister_client(websocket)

    def start_streaming(self) -> None:
        """Start the data streaming service."""
        if self.streaming_active:
            logger.warning("Streaming already active")
            return

        self.streaming_active = True
        self._stop_event.clear()

        logger.info("Starting WebSocket streaming service")

    def stop_streaming(self) -> None:
        """Stop the data streaming service."""
        if not self.streaming_active:
            return

        logger.info("Stopping WebSocket streaming service")
        self.streaming_active = False
        self._stop_event.set()


class WSSSecurity:
    """WebSocket Security features for WSS connections."""

    @staticmethod
    def create_ssl_context(ssl_config: SSLConfig) -> ssl.SSLContext:
        """Create SSL context for WebSocket Secure (WSS)."""
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

        # Load certificate and key
        context.load_cert_chain(ssl_config.cert_file, ssl_config.key_file)

        # Security settings
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.set_ciphers(ssl_config.ciphers)

        # Additional security options
        context.options |= ssl.OP_NO_SSLv2
        context.options |= ssl.OP_NO_SSLv3
        context.options |= ssl.OP_NO_TLSv1
        context.options |= ssl.OP_NO_TLSv1_1
        context.options |= ssl.OP_SINGLE_DH_USE
        context.options |= ssl.OP_SINGLE_ECDH_USE

        return context

    @staticmethod
    def validate_origin(origin: str, allowed_origins: list[str]) -> bool:
        """Validate WebSocket origin for security."""
        if not origin:
            return False

        # Check against allowed origins
        for allowed in allowed_origins:
            if origin == allowed or origin.endswith(allowed):
                return True

        return False


async def run_websocket_server(
    host: str = "0.0.0.0",
    port: int | None = None,
    ssl_config: SSLConfig | None = None,
) -> None:
    """Run WebSocket server with SSL support."""
    # Load SSL configuration
    if ssl_config is None:
        ssl_config = load_ssl_config()

    # Use configured port or default
    if port is None:
        port = ssl_config.wss_port_streaming if ssl_config else 8001

    # Initialize data stream manager
    stream_manager = DataStreamManager(ssl_config)

    # Origin validation for security
    allowed_origins = (
        [
            f"https://{ssl_config.domain}:{ssl_config.https_port_frontend}",
            f"https://localhost:{ssl_config.https_port_frontend}",
            f"https://127.0.0.1:{ssl_config.https_port_frontend}",
            "https://localhost:8444",
            "https://127.0.0.1:8444",
        ]
        if ssl_config
        else ["http://localhost:8501"]
    )

    def check_origin(origin):
        """Origin validation function."""
        return WSSSecurity.validate_origin(origin, allowed_origins)

    # SSL context for WSS
    ssl_context = None
    protocol = "ws"

    if (
        ssl_config
        and Path(ssl_config.cert_file).exists()
        and Path(ssl_config.key_file).exists()
    ):
        ssl_context = WSSSecurity.create_ssl_context(ssl_config)
        protocol = "wss"
        logger.info(f"Starting WSS server on {host}:{port}")
        logger.info(f"Certificate: {ssl_config.cert_file}")
        logger.info(f"Key: {ssl_config.key_file}")
        logger.info(f"Allowed origins: {allowed_origins}")
    else:
        logger.warning(
            "SSL certificates not found, starting WebSocket server without SSL",
        )
        logger.warning("For production, please setup SSL certificates")

    # Start data polling in background
    stream_manager.start_streaming()

    # Create server
    server = await websockets.serve(
        stream_manager.client_handler,
        host,
        port,
        ssl=ssl_context,
        origins=allowed_origins,
        ping_interval=30,  # Send ping every 30 seconds
        ping_timeout=10,  # Wait 10 seconds for pong response
        close_timeout=10,  # Wait 10 seconds for close confirmation
        max_size=2**20,  # 1MB max message size
        max_queue=32,  # Max queued messages per connection
    )

    logger.info(f"WebSocket server started: {protocol}://{host}:{port}")

    # Start data polling loop
    data_task = asyncio.create_task(stream_manager.data_polling_loop())

    try:
        # Keep server running
        await server.wait_closed()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        # Cleanup
        stream_manager.stop_streaming()
        data_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await data_task

        server.close()
        await server.wait_closed()
        logger.info("WebSocket server stopped")


def signal_handler(signum, frame) -> None:
    """Handle shutdown signals gracefully."""
    logger.info(f"Received signal {signum}, shutting down...")
    # The asyncio event loop will handle the KeyboardInterrupt


def main() -> None:
    """Main entry point for WebSocket streaming service."""
    import argparse

    parser = argparse.ArgumentParser(
        description="MFC Real-time WebSocket Streaming Service",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument(
        "--port",
        type=int,
        help="Port to bind to (default from SSL config)",
    )
    parser.add_argument(
        "--init-ssl",
        action="store_true",
        help="Initialize SSL certificates",
    )
    parser.add_argument("--test-ssl", action="store_true", help="Test SSL connection")

    args = parser.parse_args()

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    if args.init_ssl:
        logger.info("Initializing SSL infrastructure...")
        success, config = initialize_ssl_infrastructure()
        if success:
            logger.info("✅ SSL infrastructure initialized")
        else:
            logger.error("❌ SSL initialization failed")
            sys.exit(1)
        return

    if args.test_ssl:
        ssl_config = load_ssl_config()
        if ssl_config:
            port = args.port or ssl_config.wss_port_streaming
            success = test_ssl_connection(ssl_config.domain, port)
            if success:
                logger.info("✅ SSL connection test passed")
            else:
                logger.error("❌ SSL connection test failed")
                sys.exit(1)
        else:
            logger.error("No SSL configuration found")
            sys.exit(1)
        return

    # Run the WebSocket server
    try:
        asyncio.run(run_websocket_server(host=args.host, port=args.port))
    except KeyboardInterrupt:
        logger.info("WebSocket server stopped by user")
    except Exception as e:
        logger.exception(f"WebSocket server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
