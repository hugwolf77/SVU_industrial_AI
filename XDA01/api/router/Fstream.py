
# import logging
import os
from datetime import datetime
from pytz import timezone
import json
import random
from uuid import uuid1
from pprint import pprint

import asyncio
# from contextlib import asynccontextmanager

from fastapi import APIRouter, File, UploadFile
from fastapi.requests import Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from typing import Annotated

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

Fstream = APIRouter(
    prefix="/Fstream",
    tags=['Fstream'],
    responses={404:{"description":"Not found"}},
)
Fstream.mount("/static", StaticFiles(directory="api/static"), name="static")

@Fstream.get("/", response_class=HTMLResponse) # Route Path
async def DLms_branch_home(request: Request):
    client_ip = request.client.host
    # logger.warning("Client %s connected", client_ip)
    return templates.TemplateResponse("index_Fstream.html",{"request":request})

@Fstream.post("/file/")
async def create_file(file: Annotated[bytes, File()]):
    return {"file_size": len(file)}

@Fstream.post("/uploadfiles/")
async def create_upload_files(file: UploadFile = File(description="UploadFile"),
):
# async def create_upload_files(files: list[UploadFile] = File(description="Multiple files as UploadFile"),):
    UPLOAD_DIR = "./DB/storage"
    print(file.filename)
    content = await file.read()
    sFilename = f"{str(datetime.now(timezone('asia/seoul')).strftime('%Y%m%d_%H%M'))}_{file.filename}"
    with open(os.path.join(UPLOAD_DIR, sFilename), "wb") as fp:
        fp.write(content) 
    # for f in files:
    #     print(f.filename)
    #     content = await f.read()
    #     sFilename = f"{str(datetime.now(timezone('asia/seoul')).strftime('%Y%m%d_%H%M'))}_{f.filename}.csv"
    #     with open(os.path.join(UPLOAD_DIR, sFilename), "wb") as fp:
    #        fp.write(content) 
    return {"filename": sFilename}





