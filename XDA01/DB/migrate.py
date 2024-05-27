
import os
from config import Settings
from DB.DBmodel.dataTB import Base, ETT_H_1, ETT_H_2, ETT_M_1, ETT_M_2 

from sqlalchemy.orm import Session
from sqlalchemy import select

import pandas as pd
from DB.connect import get_session, conn_DB

db_con = conn_DB()
print(db_con)

def init_DB():
    # DB 초기화 : 테이블 생성
    Base.metadata.create_all(db_con)

def rawData(dataTags):
    # raw data Tag : ex 'ETT'
    filePath = os.path.join(os.getcwd()+dataTags['ETT'])
    fList = [file for file in os.listdir(filePath) if os.path.splitext(file)[1] == '.csv'  ] 
    colname = [ #Index,
                    'date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT', ]
    dfList = []
    for file in fList:
        df = pd.read_csv(os.path.join(filePath,file))
        df.columns = colname
        dfList.append(df)
    return dfList

def migrate(nameTB_mbr:str, migData:pd.DataFrame, if_exists = 'replace'): # 'replace', 'fail'
    # 데이터 입력
    # nameTB_mbr : 'ETT_H_1'
    with db_con.connect() as con:
        migData.to_sql(
        name=nameTB_mbr.upper(),
        con=con,
        if_exists=if_exists
        )

def comfirmTB(TB,iDate):
    # 입력 확인
    session = Session(db_con)
    stmt = select(TB).where(TB.date.in_([iDate]))
    for Item in session.scalars(stmt):
        print(Item)
    session.close()

if __name__=="__main__":
    # DataBase initiation
    init_DB()

    # raw data file load to DataFrame
    dataTags = {'ETT':'DB/storage/ETT/'}
    dfList = rawData(dataTags)

    # migration rawData to DB Table
    nameTB =  ['ETT_H_1', 'ETT_H_2', 'ETT_M_1', 'ETT_M_2' ]
    for df in zip(nameTB,dfList):
        migrate(df[0], df[1])

    # comfirm migrated Table
    migTB = [ETT_H_1, ETT_H_2, ETT_M_1, ETT_M_2 ]
    comfirmTB(migTB[0])

    # Base.metadata.bind(db_con)
    # asyncio.run(migrate_table())
   
