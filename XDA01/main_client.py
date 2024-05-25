# application main.py

import logging
from client.app import appRun
from config import Settings

from pytz import timezone
from datetime import datetime
today = datetime.now(timezone('Asia/Seoul'))

# from DB import migrate as mig
# from DB.DBmodel.dataTB import Base, ETT_H_1, ETT_H_2, ETT_M_1, ETT_M_2 

# configs = Settings()

# service_url = configs.SERVICE_URL
# service_port = configs.SERVICE_PORT

# logging
logging.basicConfig(
    filename="./logging.log",
    #stream=sys.stdout, 
    level=logging.INFO, 
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    appRun()
