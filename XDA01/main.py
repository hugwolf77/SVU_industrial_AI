#
import logging
from client.app import appRun
from api.service import apiRun
from config import Settings

from pytz import timezone
from datetime import datetime
today = datetime.now(timezone('Asia/Seoul'))


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
    appRun()