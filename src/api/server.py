from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
from src.api.state import DashboardState

app = FastAPI(title="Strategist Hub API")

# Enable CORS for React development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

state_manager = DashboardState()

@app.get("/api/state")
async def get_state():
    """Return the current full system state."""
    return state_manager.get_full_state()

@app.get("/api/history")
async def get_history():
    """Read trade history from CSV."""
    import pandas as pd
    hist_path = "logs/trade_history.csv"
    if os.path.exists(hist_path):
        df = pd.read_csv(hist_path)
        return df.to_dict('records')
    return []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time update stream."""
    await websocket.accept()
    try:
        while True:
            # Broadcast the state every second
            await websocket.send_json(state_manager.get_full_state())
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass

if __name__ == "__main__":
    import uvicorn
    import os
    uvicorn.run(app, host="0.0.0.0", port=8000)
