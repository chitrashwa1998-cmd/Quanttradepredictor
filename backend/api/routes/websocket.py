"""
WebSocket API endpoints
Handles real-time data streaming and live predictions
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
import json
import asyncio
import logging
from typing import Dict, Any, Set
import sys
import os

# Add parent directory to path to import existing utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# from utils.live_prediction_pipeline import LivePredictionPipeline
# from utils.upstox_websocket import UpstoxWebSocketManager
from core.database import get_database

logger = logging.getLogger(__name__)
router = APIRouter()

class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.live_pipeline = None
        self.websocket_manager = None
    
    async def connect(self, websocket: WebSocket):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"New WebSocket connection. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
            
        message_str = json.dumps(message)
        disconnected = set()
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.error(f"Failed to send message to WebSocket: {e}")
                disconnected.add(connection)
        
        # Remove disconnected connections
        for conn in disconnected:
            self.active_connections.discard(conn)
    
    async def start_live_pipeline(self):
        """Start live prediction pipeline"""
        if self.live_pipeline is None:
            try:
                # TODO: Initialize LivePredictionPipeline when available
                self.live_pipeline = {"status": "mock"}  # Mock for now
                
                # Start pipeline in background
                asyncio.create_task(self._run_live_pipeline())
                logger.info("✅ Live prediction pipeline started")
                
            except Exception as e:
                logger.error(f"❌ Failed to start live pipeline: {e}")
                raise
    
    async def _run_live_pipeline(self):
        """Background task for live predictions"""
        while self.active_connections and self.live_pipeline:
            try:
                # Get latest predictions (mock for now)
                predictions = {"timestamp": "2025-01-01T00:00:00", "volatility": 0.15}
                
                if predictions:
                    await self.broadcast({
                        "type": "predictions",
                        "data": predictions,
                        "timestamp": predictions.get('timestamp')
                    })
                
                # Wait before next update
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Live pipeline error: {e}")
                await asyncio.sleep(10)  # Wait longer on error
    
    async def stop_live_pipeline(self):
        """Stop live prediction pipeline"""
        if self.live_pipeline:
            # TODO: Cleanup when real pipeline is implemented
            self.live_pipeline = None
            logger.info("Live prediction pipeline stopped")

# Global connection manager
manager = ConnectionManager()

@router.websocket("/live-predictions")
async def websocket_live_predictions(websocket: WebSocket):
    """WebSocket endpoint for live predictions"""
    await manager.connect(websocket)
    
    try:
        # Start live pipeline if not already running
        await manager.start_live_pipeline()
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for client messages (ping/pong, commands, etc.)
                message = await websocket.receive_text()
                data = json.loads(message)
                
                # Handle different message types
                if data.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                
                elif data.get("type") == "request_status":
                    # Send current status
                    status = {
                        "type": "status",
                        "connected": True,
                        "live_pipeline": manager.live_pipeline is not None,
                        "connections": len(manager.active_connections)
                    }
                    await websocket.send_text(json.dumps(status))
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket message handling error: {e}")
                break
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        manager.disconnect(websocket)
        
        # Stop pipeline if no more connections
        if not manager.active_connections:
            await manager.stop_live_pipeline()

@router.websocket("/market-data")
async def websocket_market_data(websocket: WebSocket):
    """WebSocket endpoint for raw market data"""
    await manager.connect(websocket)
    
    try:
        # Initialize market data WebSocket (mock for now)
        if not manager.websocket_manager:
            try:
                # TODO: Initialize UpstoxWebSocketManager when available
                manager.websocket_manager = {"status": "mock"}
                
                # Send mock data for demonstration
                asyncio.create_task(_send_mock_market_data())
                
            except Exception as e:
                logger.warning(f"Market data WebSocket not available: {e}")
                # Send mock data for demonstration
                asyncio.create_task(_send_mock_market_data())
        
        # Keep connection alive
        while True:
            try:
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if data.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
                
            except WebSocketDisconnect:
                break
    
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)

async def _send_mock_market_data():
    """Send mock market data when real data is not available"""
    import random
    import pandas as pd
    
    while manager.active_connections:
        try:
            # Generate mock NIFTY 50 data
            mock_data = {
                "symbol": "NIFTY50",
                "price": round(23000 + random.uniform(-100, 100), 2),
                "change": round(random.uniform(-50, 50), 2),
                "volume": random.randint(1000000, 5000000),
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            await manager.broadcast({
                "type": "market_data",
                "data": mock_data,
                "timestamp": mock_data["timestamp"]
            })
            
            await asyncio.sleep(1)  # Update every second
            
        except Exception as e:
            logger.error(f"Mock market data error: {e}")
            await asyncio.sleep(5)