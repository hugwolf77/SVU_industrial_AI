
import json
import logging
import random
from datetime import datetime

import asyncio
from contextlib import asynccontextmanager

from fastapi import APIRouter
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from api.model.dataclass import DataInput, PredictOutput

from MLmodels.DLM import DLM

logger = logging.getLogger("service_DMms")

templates = Jinja2Templates(directory="api/templates")

mlModel= {}

@asynccontextmanager
async def lifespan(app: APIRouter, NM):
    # Load the ML model
    mlModel = DLM().predict
    yield
    mlModel.clear()

DLms = APIRouter(
    prefix="/DLM",
    tags=['DLM'],
    responses={404:{"description":"Not found"}},
    lifespan=lifespan
)
DLms.mount("/static", StaticFiles(directory="api/static"), name="static")

### model

random.seed()
async def generate_random_data(request: Request):
    """
    Generates random value between 0 and 100
    :return: String containing current timestamp (YYYY-mm-dd HH:MM:SS) and randomly generated data.
    """
    while True:
        json_data = json.dumps(
            {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "value": random.random() * 100,
            }
        )
        yield f"data:{json_data}\n\n"
        # yield f"{json_data}\n\n"
        await asyncio.sleep(1)

@DLms.get("/", response_class=HTMLResponse) # Route Path
async def DLms_branch_home(request: Request):
    client_ip = request.client.host
    logger.warning("Client %s connected", client_ip)
    return templates.TemplateResponse("index_dlm.html",{"request":request})


@DLms.get("/example", response_class=HTMLResponse) # Route Path
async def DLms_branch_home(request: Request):
    client_ip = request.client.host
    logger.info("Client %s connected", client_ip)
    return templates.TemplateResponse("DLms_home.html",{"request":request})


@DLms.get("/fakeStream", response_class=HTMLResponse) # Route Path
async def chart_data(request: Request):
    response = StreamingResponse(generate_random_data(request), media_type='text/event-stream') # application/x-ndjson
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response 


@DLms.post("/predict", tags=['DLM'], response_model=PredictOutput)
async def NN01_predict(request_input: DataInput, request: Request):
    client_ip = request.client.host
    logger.info("Client %s connected for prediction result", client_ip)
    result =  mlModel["predict"](request_input.x)
    return {'prediction' : result}