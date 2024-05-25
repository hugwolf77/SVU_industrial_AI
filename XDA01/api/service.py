# api service server for model predict 
import sys
sys.path.insert(2,'../MLmodels/DLM')
import os
import logging

from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import signal

# routing path
from .router.DLms import DLms 
from MLmodels.DLM import DLM

logger = logging.getLogger("service_DMms")

api = FastAPI()
print(f"pwd: {os.getcwd()}")
api.mount("/static", StaticFiles(directory="./api/static"), name="static")

templates = Jinja2Templates(directory="./api/templates")

api.include_router(DLms)

@api.get("/", response_class=HTMLResponse) # root
def index(request: Request):
    # return {"Python": "Framework",}
    return templates.TemplateResponse(
        request=request, name="index_root.html")

def apiRun(url, port):
    logger.info(f"!! service server start !!")
    uvicorn.run("api.service:api", host=url, port=int(port), reload=True)


def shutdown():
    os.kill(os.getpid(), signal.SIGTERM)
    logger.info(f"!! service server shutting down !!")
    return Response(status_code=200, content='Server shutting down...')

@api.on_event('shutdown')
def on_shutdown():
    logger.info(f"!! service server shutting down !!")
    print('Server shutting down...')

if __name__ == "__main__":
    apiRun('127.0.0.1',8000)