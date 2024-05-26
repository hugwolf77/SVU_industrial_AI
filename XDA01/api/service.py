# api service server for model predict 
# import logging
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# routing path
from .router.DLms import DLms 
from .router.Fstream import Fstream

# logging
# logging.basicConfig(
#     filename="./logging.log",
#     #stream=sys.stdout, 
#     level=logging.INFO, 
#     format="%(asctime)s %(levelname)s %(message)s",
#     datefmt="%m/%d/%Y %I:%M:%S %p",)
# logger = logging.getLogger(__name__)

api = FastAPI()
api.mount("/static", StaticFiles(directory="./api/static"), name="static")

templates = Jinja2Templates(directory="./api/templates")

api.include_router(DLms)
api.include_router(Fstream)

@api.get("/", response_class=HTMLResponse) # root
def index(request: Request):
    # return {"Python": "Framework",}
    return templates.TemplateResponse(
        request=request, name="index_root.html")

def apiRun(url, port):
    # logger.info(f"!! service server start !!")
    uvicorn.run("api.service:api", host=url, port=int(port), reload=True)


# def shutdown():
#     os.kill(os.getpid(), signal.SIGTERM)
#     logger.info(f"!! service server shutting down !!")
#     return Response(status_code=200, content='Server shutting down...')

# @api.on_event('shutdown')
# def on_shutdown():
#     logger.info(f"!! service server shutting down !!")
#     print('Server shutting down...')

if __name__ == "__main__":
    apiRun('127.0.0.1',8000)