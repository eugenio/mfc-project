"""
Real-time Data Streaming Service for MFC Monitoring

Provides WebSocket-based real-time data streaming, event handling,
and distributed monitoring capabilities for MFC systems.
"""
import asyncio
import websockets
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Set, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import threading
import queue
import numpy as np
from pathlib import Path
import sys
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamEventType(Enum):
    """Types of streaming events"""
    METRICS_UPDATE = "metrics_update"
    ALERT = "alert"
    STATUS_CHANGE = "status_change"
    COMMAND_RESULT = "command_result"
    SENSOR_DATA = "sensor_data"
    CONTROL_UPDATE = "control_update"

class StreamEvent:
    """Stream event data structure"""
    event_id: str
    event_type: StreamEventType
    timestamp: datetime
    data: Dict[str, Any]
    priority: int = 1  # 1=highest, 10=lowest
    client_filter: Optional[List[str]] = None  # Specific clients
class ClientConnection:
    """Client connection information"""
    client_id: str
    websocket: websockets.WebSocketServerProtocol
    connected_at: datetime
    subscriptions: Set[StreamEventType]
    last_ping: datetime
    is_authenticated: bool = False
    user_role: str = "observer"  # observer, operator, admin
class RealTimeStreamer:
    """
    Real-time data streaming service for MFC monitoring
    
    Features:
    - WebSocket-based client connections
    - Event subscription management  
    - Real-time data broadcasting
    - Client authentication and authorization
    - Connection health monitoring
    """
    
    def __init__(self, host: str = "localhost", port: int = 8001):
        self.host = host
        self.port = port
        self.clients: Dict[str, ClientConnection] = {}
        self.event_queue = asyncio.Queue()
        self.mfc_model: Optional[IntegratedMFCModel] = None
        self.data_stream: Optional[MFCDataStream] = None
        self.is_running = False
        
        # Statistics
        self.stats = {
            "total_connections": 0,
            "events_sent": 0,
            "bytes_transmitted": 0,
            "start_time": datetime.now()
        }
    
    async def start_server(self):
        """Start the streaming server"""
        logger.info(f"Starting real-time streaming server on {self.host}:{self.port}")
        
        # Initialize MFC model for data generation
        self.mfc_model = IntegratedMFCModel(
            n_cells=5,
            species="mixed", 
            substrate="lactate",
            use_gpu=True,
            simulation_hours=1000
        )
        
        # Initialize data stream
        self.data_stream = MFCDataStream(
            buffer_size=10000,
            sampling_rate_hz=10.0
        )
        
        self.is_running = True
        
        # Start background tasks
        asyncio.create_task(self.data_generation_loop())
        asyncio.create_task(self.event_broadcaster())
        asyncio.create_task(self.health_monitor())
        
        # Start WebSocket server
        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10,
            max_size=1024*1024  # 1MB max message size
        ):
            logger.info("Streaming server started successfully")
            await asyncio.Future()  # Run forever
    
    async def handle_client(self, websocket, path):
        """Handle new client connection"""
        client_id = str(uuid.uuid4())
        
        try:
            # Create client connection
            client = ClientConnection(
                client_id=client_id,
                websocket=websocket,
                connected_at=datetime.now(),
                subscriptions=set(),
                last_ping=datetime.now()
            )
            
            self.clients[client_id] = client
            self.stats["total_connections"] += 1
            
            logger.info(f"Client {client_id} connected from {websocket.remote_address}")
            
            # Send welcome message
            await self.send_to_client(client_id, {
                "type": "connection_established",
                "client_id": client_id,
                "server_time": datetime.now().isoformat(),
                "available_subscriptions": [e.value for e in StreamEventType]
            })
            
            # Handle client messages
            async for message in websocket:
                await self.handle_client_message(client_id, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
        finally:
            # Clean up client
            if client_id in self.clients:
                del self.clients[client_id]
    
    async def handle_client_message(self, client_id: str, message: str):
        """Handle message from client"""
        try:
            data = json.loads(message)
            msg_type = data.get("type")
            
            if msg_type == "subscribe":
                # Subscribe to event types
                event_types = data.get("events", [])
                await self.subscribe_client(client_id, event_types)
                
            elif msg_type == "unsubscribe":
                # Unsubscribe from event types
                event_types = data.get("events", [])
                await self.unsubscribe_client(client_id, event_types)
                
            elif msg_type == "ping":
                # Update last ping time
                if client_id in self.clients:
                    self.clients[client_id].last_ping = datetime.now()
                await self.send_to_client(client_id, {"type": "pong"})
                
            elif msg_type == "authenticate":
                # Handle authentication
                await self.authenticate_client(client_id, data.get("credentials", {}))
                
            elif msg_type == "command":
                # Handle control commands (if authorized)
                await self.handle_command(client_id, data.get("command"))
                
        except json.JSONDecodeError:
            await self.send_to_client(client_id, {
                "type": "error",
                "message": "Invalid JSON format"
            })
        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}")
            await self.send_to_client(client_id, {
                "type": "error", 
                "message": str(e)
            })
    
    async def subscribe_client(self, client_id: str, event_types: List[str]):
        """Subscribe client to event types"""
        if client_id not in self.clients:
            return
            
        client = self.clients[client_id]
        
        for event_type_str in event_types:
            try:
                event_type = StreamEventType(event_type_str)
                client.subscriptions.add(event_type)
            except ValueError:
                logger.warning(f"Invalid event type: {event_type_str}")
        
        await self.send_to_client(client_id, {
            "type": "subscription_updated",
            "subscriptions": [e.value for e in client.subscriptions]
        })
    
    async def unsubscribe_client(self, client_id: str, event_types: List[str]):
        """Unsubscribe client from event types"""
        if client_id not in self.clients:
            return
            
        client = self.clients[client_id]
        
        for event_type_str in event_types:
            try:
                event_type = StreamEventType(event_type_str)
                client.subscriptions.discard(event_type)
            except ValueError:
                logger.warning(f"Invalid event type: {event_type_str}")
        
        await self.send_to_client(client_id, {
            "type": "subscription_updated",
            "subscriptions": [e.value for e in client.subscriptions]
        })
    
    async def authenticate_client(self, client_id: str, credentials: Dict[str, Any]):
        """Authenticate client (simplified for demo)"""
        if client_id not in self.clients:
            return
        
        # Simple authentication - in production, use proper auth
        username = credentials.get("username", "")
        password = credentials.get("password", "")
        
        if username == "admin" and password == "admin":
            self.clients[client_id].is_authenticated = True
            self.clients[client_id].user_role = "admin"
            auth_result = {"authenticated": True, "role": "admin"}
        elif username == "operator" and password == "operator":
            self.clients[client_id].is_authenticated = True
            self.clients[client_id].user_role = "operator"
            auth_result = {"authenticated": True, "role": "operator"}
        else:
            auth_result = {"authenticated": False, "role": "observer"}
        
        await self.send_to_client(client_id, {
            "type": "authentication_result",
            **auth_result
        })
    
    async def handle_command(self, client_id: str, command: Dict[str, Any]):
        """Handle control command from client"""
        if client_id not in self.clients:
            return
        
        client = self.clients[client_id]
        
        # Check authorization
        if not client.is_authenticated or client.user_role not in ["operator", "admin"]:
            await self.send_to_client(client_id, {
                "type": "command_result",
                "success": False,
                "error": "Insufficient privileges"
            })
            return
        
        # Process command
        try:
            cmd_type = command.get("type")
            
            if cmd_type == "start_simulation":
                # Start simulation
                result = {"success": True, "message": "Simulation started"}
            elif cmd_type == "stop_simulation":
                # Stop simulation
                result = {"success": True, "message": "Simulation stopped"}
            elif cmd_type == "emergency_stop":
                # Emergency stop
                result = {"success": True, "message": "Emergency stop activated"}
            else:
                result = {"success": False, "error": f"Unknown command: {cmd_type}"}
            
            await self.send_to_client(client_id, {
                "type": "command_result",
                **result
            })
            
        except Exception as e:
            await self.send_to_client(client_id, {
                "type": "command_result",
                "success": False,
                "error": str(e)
            })
    
    async def send_to_client(self, client_id: str, data: Dict[str, Any]):
        """Send data to specific client"""
        if client_id not in self.clients:
            return
        
        try:
            message = json.dumps(data, default=str)
            await self.clients[client_id].websocket.send(message)
            self.stats["events_sent"] += 1
            self.stats["bytes_transmitted"] += len(message)
            
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} connection closed")
            if client_id in self.clients:
                del self.clients[client_id]
        except Exception as e:
            logger.error(f"Error sending to client {client_id}: {e}")
    
    async def broadcast_event(self, event: StreamEvent):
        """Broadcast event to subscribed clients"""
        if not self.clients:
            return
        
        message = {
            "event_id": event.event_id,
            "type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "data": event.data
        }
        
        # Send to subscribed clients
        for client_id, client in self.clients.items():
            if event.event_type in client.subscriptions:
                # Check client filter if specified
                if event.client_filter and client_id not in event.client_filter:
                    continue
                
                await self.send_to_client(client_id, message)
    
    async def add_event(self, event_type: StreamEventType, data: Dict[str, Any], 
                       priority: int = 1, client_filter: List[str] = None):
        """Add event to broadcast queue"""
        event = StreamEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(),
            data=data,
            priority=priority,
            client_filter=client_filter
        )
        
        await self.event_queue.put(event)
    
    async def event_broadcaster(self):
        """Background task to broadcast events"""
        while self.is_running:
            try:
                # Get event from queue (with timeout)
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self.broadcast_event(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in event broadcaster: {e}")
    
    async def data_generation_loop(self):
        """Generate simulated real-time data"""
        while self.is_running:
            try:
                # Generate metrics data
                current_time = datetime.now()
                
                # Simulate MFC metrics
                metrics_data = {
                    "timestamp": current_time.isoformat(),
                    "power_output_w": 5.0 + np.random.normal(0, 0.5),
                    "efficiency_pct": 80.0 + np.random.normal(0, 5),
                    "temperature_c": 25.0 + np.random.normal(0, 1),
                    "ph_level": 7.0 + np.random.normal(0, 0.2),
                    "pressure_bar": 1.0 + np.random.normal(0, 0.05),
                    "flow_rate_ml_min": 100.0 + np.random.normal(0, 5),
                    "cell_voltages": [0.7 + np.random.normal(0, 0.02) for _ in range(5)],
                    "current_densities": [5.0 + np.random.normal(0, 0.5) for _ in range(5)],
                    "biofilm_thickness": [10.0 + np.random.normal(0, 1) for _ in range(5)],
                    "substrate_concentrations": [50.0 + np.random.normal(0, 2) for _ in range(5)]
                }
                
                # Add metrics event
                await self.add_event(StreamEventType.METRICS_UPDATE, metrics_data)
                
                # Occasionally generate alerts
                if np.random.random() < 0.1:  # 10% chance
                    alert_data = {
                        "level": "warning",
                        "message": f"Temperature spike detected: {metrics_data['temperature_c']:.1f}°C",
                        "category": "temperature",
                        "timestamp": current_time.isoformat()
                    }
                    await self.add_event(StreamEventType.ALERT, alert_data)
                
            except Exception as e:
                logger.error(f"Error in data generation: {e}")
            
            await asyncio.sleep(1.0)  # Update every second
    
    async def health_monitor(self):
        """Monitor client connections and system health"""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                # Check for stale connections
                stale_clients = []
                for client_id, client in self.clients.items():
                    if (current_time - client.last_ping).total_seconds() > 60:
                        stale_clients.append(client_id)
                
                # Remove stale clients
                for client_id in stale_clients:
                    logger.info(f"Removing stale client: {client_id}")
                    del self.clients[client_id]
                
                # Send health status
                health_data = {
                    "active_clients": len(self.clients),
                    "uptime_seconds": (current_time - self.stats["start_time"]).total_seconds(),
                    "events_sent": self.stats["events_sent"],
                    "bytes_transmitted": self.stats["bytes_transmitted"]
                }
                
                await self.add_event(StreamEventType.STATUS_CHANGE, health_data)
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
            
            await asyncio.sleep(30.0)  # Check every 30 seconds
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server statistics"""
        current_time = datetime.now()
        uptime = current_time - self.stats["start_time"]
        
        return {
            "active_clients": len(self.clients),
            "total_connections": self.stats["total_connections"],
            "events_sent": self.stats["events_sent"],
            "bytes_transmitted": self.stats["bytes_transmitted"],
            "uptime_seconds": uptime.total_seconds(),
            "start_time": self.stats["start_time"].isoformat(),
            "is_running": self.is_running
        }
