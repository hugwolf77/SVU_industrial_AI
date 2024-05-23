import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="./.env",verbose=True)

class Settings:
    # Service API Info
    SERVICE_URL = os.getenv("SERVICE_URL")
    SERVICE_PORT = os.getenv("SERVICE_PORT")

    # DataBase Info
    DB_USERNAME : str = os.getenv("DB_USERNAME")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_HOST : str = os.getenv("DB_HOST")
    DB_PORT : str = os.getenv("DB_PORT")
    DB_DATABASE : str = os.getenv("DB_DATABASE")
    DATABASE_URL = os.getenv("DATABASE_URL")
