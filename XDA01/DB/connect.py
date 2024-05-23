
from sqlalchemy import create_engine, MetaData, Table, Column
from sqlalchemy.orm import sessionmaker
from config import Settings

configs = Settings()
SQLALCHEMY_DATABASE_URL = configs.DATABASE_URL 

engine = create_engine(SQLALCHEMY_DATABASE_URL, echo=True) 
SessionLocal = sessionmaker(autocommit=False,autoflush=False,bind=engine)

def conn_DB():
    db_conn = engine
    return db_conn

def get_session():
	session = SessionLocal()
	try:
		yield session # DB 연결 성공한 경우, DB 세션 시작
	finally:
		session.close()
		# db 세션이 시작된 후, API 호출이 마무리되면 DB 세션을 닫아준다.
