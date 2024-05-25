# application main.py


import logging
from api.service import apiRun, shutdown
from config import Settings

from pytz import timezone
from datetime import datetime
today = datetime.now(timezone('Asia/Seoul'))

from DB import migrate as mig
from DB.DBmodel.dataTB import Base, ETT_H_1, ETT_H_2, ETT_M_1, ETT_M_2 

configs = Settings()

service_url = configs.SERVICE_URL
service_port = configs.SERVICE_PORT

# logging
logging.basicConfig(
    filename="./logging.log",
    #stream=sys.stdout, 
    level=logging.INFO, 
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    apiRun(service_url,service_port)

    ## init DataBase
    # mig.init_DB()
    
    ## raw data file load to DataFrame
    # dataTags = {'ETT':'/DB/storage/ETDataset-raw/'}
    # dfList = mig.rawData(dataTags)

    # # migration rawData to DB Table
    # nameTB =  ['ETT_H_1', 'ETT_H_2', 'ETT_M_1', 'ETT_M_2' ]
    # for df in zip(nameTB,dfList):
    #     mig.migrate(df[0], df[1])

    # # comfirm migrated Table
    # migTB = [ETT_H_1, ETT_H_2, ETT_M_1, ETT_M_2 ]
    # iDate = '2016-07-01 01:00:00'
    # mig.comfirmTB(migTB[0],iDate)