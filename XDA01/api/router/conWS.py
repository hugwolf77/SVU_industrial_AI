import os
from typing import Annotated, Union
from fastapi import (
    Cookie,
    Depends,
    APIRouter,
    Query,
    WebSocket,
    WebSocketException,
    WebSocketDisconnect,
    status
)

from fastapi.responses import HTMLResponse


filePath = './API_app/templates/Chat_03.html'
with open(filePath, 'r') as f:
    html = f.read()

conWS = APIRouter(
    prefix="/conWS", # url 앞에 고정적으로 붙는 prefix path 추가
    tags=['conWS'],
    responses={404:{"description":"Not found"}},
) 

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
        
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()


@conWS.get("/")
async def get():
    return HTMLResponse(html)

# simple ws chat
# @conWS.websocket("/ws")
# async def wevsocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     while True:
#         data = await websocket.receive_text()
#         await websocket.send_text(f"message text was: {data}")


# parameter add chat
# async def get_cookie_or_token(
#     websocket: WebSocket, 
#     session: Annotated[Union[str, None], Cookie()] = None,
#     token: Annotated[Union[str,None], Query()] = None,
# ):
#     if session is None and token is None:
#         raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)
#     return session or token

# @conWS.websocket("/items/{item_id}/ws")
# async def websocket_endpoint(
#     *,
#     websocket: WebSocket,
#     item_id: str,
#     q: Union[int, None] = None,
#     cookie_or_token: Annotated[str, Depends(get_cookie_or_token)],
# ):
#     await websocket.accept()
#     while True:
#         data = await websocket.receive_text()
#         await websocket.send_text(
#             f"Session cookie or query token value is: {cookie_or_token}"
#         )
#         if q is not None:
#             await websocket.send_text(f"Query parameter q is: {q}")
#         await websocket.send_text(f"Message text was: {data}, for item ID: {item_id}")

# multi chat

@conWS.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message(f"You wrote: {data}", websocket)
            await manager.broadcast(f"Client #{client_id} says: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client #{client_id} left the chat")    
        