
# import logging
from datetime import datetime
from pytz import timezone
import json
import random

import asyncio
# from contextlib import asynccontextmanager

from fastapi import APIRouter
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from api.model.dataclass import DataInput #, PredictOutput # data type validation too hard :(

import torch
from MLmodels.DLM import DLM
import pandas as pd
from config import Modelargs

# logger = logging.getLogger("service_DMms")

templates = Jinja2Templates(directory="api/templates")

args = Modelargs()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

mlModel= {
    "DLM" : DLM(args)
}

# @asynccontextmanager
# async def lifespan(DLms: APIRouter):
#     # Load the ML model
#     mlModel["DLM"] = DLM(args)
#     print(mlModel)
#     yield
#     mlModel.clear()

DLms = APIRouter(
    prefix="/DLM",
    tags=['DLM'],
    responses={404:{"description":"Not found"}},
    # lifespan=lifespan
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
                "time": datetime.now(timezone('asia/seoul')).strftime("%Y-%m-%d %H:%M:%S"),
                "value": random.random() * 100,
            }
        )
        yield f"data:{json_data}\n\n"
        # yield f"{json_data}\n\n"
        await asyncio.sleep(1)

@DLms.get("/", response_class=HTMLResponse) # Route Path
async def DLms_branch_home(request: Request):
    client_ip = request.client.host
    # logger.warning("Client %s connected", client_ip)
    return templates.TemplateResponse("index_dlm.html",{"request":request})


@DLms.get("/example", response_class=HTMLResponse) # Route Path
async def DLms_branch_home(request: Request):
    client_ip = request.client.host
    # logger.info("Client %s connected", client_ip)
    return templates.TemplateResponse("DLms_home.html",{"request":request})


@DLms.get("/fakeStream", response_class=HTMLResponse) # Route Path
async def chart_data(request: Request):
    response = StreamingResponse(generate_random_data(request), media_type='text/event-stream') # application/x-ndjson
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response 

@DLms.post("/predict", tags=['DLM'], response_model=dict)
async def DLM_predict(request_input: DataInput, request: Request):
    client_ip = request.client.host
    # logger.info("Client %s connected for prediction result", client_ip)
    data = {
         'date' : request_input.date, 
         'HUFL' : request_input.HUFL, 
         'HULL' : request_input.HULL, 
         'MUFL' : request_input.MUFL, 
         'MULL' : request_input.MULL, 
         'LUFL' : request_input.LUFL, 
         'LULL' : request_input.LULL, 
         'OT' : request_input.OT,
    }
    args.direct_pred_input = pd.DataFrame(data)
    # print(args.direct_pred_input.shape)
    setting = '{}_{}_{}_ft{}_sl{}_pl{}_eb{}_{}'.format(args.model_id, args.model, args.data, args.features, args.seq_len,args.pred_len,args.embed,args.des)
    result =  mlModel["DLM"].predict(setting,load=True)
    # print(f"result:{result}")
    respons_Result = {
         'date' : list(result.date.astype(str)), 
         'HUFL' : list(result.HUFL), 
         'HULL' : list(result.HULL), 
         'MUFL' : list(result.MUFL), 
         'MULL' : list(result.MULL), 
         'LUFL' : list(result.LUFL), 
         'LULL' : list(result.LULL), 
         'OT'   : list(result.OT),
    }

    # respons_Result =PredictOutput(
    #      date = list(result.date.astype(str)), 
    #      HUFL = list(result.HUFL.astype(float)), 
    #      HULL = list(result.HULL.astype(float)), 
    #      MUFL = list(result.MUFL.astype(float)), 
    #      MULL = list(result.MULL.astype(float)), 
    #      LUFL = list(result.LUFL.astype(float)), 
    #      LULL = list(result.LULL.astype(float)), 
    #      OT   = list(result.OT.astype(float)),
    # )
    # respons_json = json.dumps(respons_Result)

    return {'prediction': respons_Result}