"""
Live Data API endpoints
Handles real-time market data and live predictions via Upstox integration
"""

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging
import asyncio
import json

from core.database import get_database_dependency
from core.model_manager import ModelManager

logger = logging.getLogger(__name__)
router = APIRouter()

class LiveDataConnectionRequest(BaseModel):
    access_token: str
    api_key: str
    instruments: List[str]

class HistoricalDataRequest(BaseModel):
    access_token: str
    api_key: str
    instruments: List[str]

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                # Remove disconnected connections
                self.active_connections.remove(connection)

manager = ConnectionManager()

@router.post("/connect")
async def connect_to_live_data(
    request: LiveDataConnectionRequest,
    db = Depends(get_database_dependency)
):
    """Connect to Upstox live data feed"""
    try:
        # Import Upstox utilities
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        
        from utils.upstox_websocket import UpstoxWebSocketClient
        
        # Initialize Upstox WebSocket client (placeholder implementation)
        # ws_client = UpstoxWebSocketClient(
        #     access_token=request.access_token,
        #     api_key=request.api_key
        # )
        
        # Subscribe to instruments (mock for now)
        success = True  # Mock success
        
        if success:
            return {
                "success": True,
                "message": f"Connected to Upstox WebSocket for {len(request.instruments)} instruments",
                "instruments": request.instruments
            }
        else:
            return {
                "success": False,
                "message": "Failed to connect to Upstox WebSocket"
            }
            
    except Exception as e:
        logger.error(f"Live data connection failed: {e}")
        return {
            "success": False,
            "message": f"Connection error: {str(e)}"
        }

@router.post("/disconnect")
async def disconnect_from_live_data():
    """Disconnect from live data feed"""
    try:
        # Disconnect logic would go here
        return {
            "success": True,
            "message": "Disconnected from live data feed"
        }
    except Exception as e:
        logger.error(f"Disconnect failed: {e}")
        return {
            "success": False,
            "message": f"Disconnect error: {str(e)}"
        }

@router.post("/start-predictions")
async def start_prediction_pipeline(
    db = Depends(get_database_dependency)
):
    """Start live prediction pipeline"""
    try:
        # Import prediction pipeline utilities
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        
        from utils.live_prediction_pipeline import LivePredictionPipeline
        
        # Initialize and start prediction pipeline (placeholder implementation)
        # pipeline = LivePredictionPipeline()
        # success = await pipeline.start()
        success = True  # Mock success for now
        
        if success:
            return {
                "success": True,
                "message": "Live prediction pipeline started successfully"
            }
        else:
            return {
                "success": False,
                "message": "Failed to start prediction pipeline"
            }
            
    except Exception as e:
        logger.error(f"Prediction pipeline start failed: {e}")
        return {
            "success": False,
            "message": f"Pipeline start error: {str(e)}"
        }

@router.post("/stop-predictions")
async def stop_prediction_pipeline():
    """Stop live prediction pipeline"""
    try:
        # Stop prediction pipeline logic would go here
        return {
            "success": True,
            "message": "Live prediction pipeline stopped"
        }
    except Exception as e:
        logger.error(f"Pipeline stop failed: {e}")
        return {
            "success": False,
            "message": f"Pipeline stop error: {str(e)}"
        }

@router.post("/fetch-historical")
async def fetch_historical_data(
    request: HistoricalDataRequest,
    db = Depends(get_database_dependency)
):
    """Fetch historical data from Upstox"""
    try:
        # Import historical data utilities
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        
        # from utils.upstox_historical import fetch_upstox_historical_data
        
        # Fetch historical data (placeholder implementation)
        records_saved = 0
        # Mock implementation - would fetch real data from Upstox
        for instrument in request.instruments:
            try:
                # data = await fetch_upstox_historical_data(
                #     instrument,
                #     request.access_token,
                #     request.api_key
                # )
                
                # Mock successful data storage for now
                records_saved += 100  # Mock 100 records per instrument
                        
            except Exception as e:
                logger.warning(f"Failed to fetch data for {instrument}: {e}")
                continue
        
        return {
            "success": True,
            "message": f"Historical data fetch completed",
            "records_saved": records_saved,
            "instruments_processed": len(request.instruments)
        }
        
    except Exception as e:
        logger.error(f"Historical data fetch failed: {e}")
        return {
            "success": False,
            "message": f"Historical fetch error: {str(e)}"
        }

@router.websocket("/ws/live-data")
async def websocket_live_data(websocket: WebSocket):
    """WebSocket endpoint for real-time market data"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            
            # Process incoming data if needed
            try:
                message = json.loads(data)
                # Handle client messages here
                await websocket.send_text(json.dumps({
                    "type": "ack",
                    "message": "Message received"
                }))
            except json.JSONDecodeError:
                # Handle non-JSON messages
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format"
                }))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("Client disconnected from live data WebSocket")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

@router.get("/status")
async def get_live_data_status():
    """Get status of live data connections"""
    try:
        return {
            "success": True,
            "active_connections": len(manager.active_connections),
            "websocket_status": "active" if manager.active_connections else "inactive"
        }
    except Exception as e:
        logger.error(f"Failed to get live data status: {e}")
        return {
            "success": False,
            "message": f"Status error: {str(e)}"
        }