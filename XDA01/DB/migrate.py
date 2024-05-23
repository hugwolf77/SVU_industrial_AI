
import os
from config import Settings
from DB.DBmodel.dataTB import Base, dataTB

from sqlalchemy.orm import Session
from sqlalchemy import select

import pandas as pd
from DB.connect import get_session, conn_DB

pwd = os.getcwd()
df_portscan = pd.read_csv(os.path.join(pwd+'/DB/Storage/data.csv'))

col_portscan = [
    # Index,                    ,
    "DESTINATION_PORT",
    "FLOW_DURATION",
    "TOTAL_FWD_PACKETS",
    "TOTAL_BACKWARD_PACKETS",
    "TOTAL_LENGTH_OF_FWD_PACKETS",
    "TOTAL_LENGTH_OF_BWD_PACKETS",
    "FWD_PACKET_LENGTH_MAX",
    "FWD_PACKET_LENGTH_MIN",
    "FWD_PACKET_LENGTH_MEAN",
    "FWD_PACKET_LENGTH_STD",
    "BWD_PACKET_LENGTH_MAX",
    "BWD_PACKET_LENGTH_MIN",
    "BWD_PACKET_LENGTH_MEAN",
    "BWD_PACKET_LENGTH_STD",
    "FLOW_BYTES_S",
    "FLOW_PACKETS_S",
    "FLOW_IAT_MEAN",
    "FLOW_IAT_STD",
    "FLOW_IAT_MAX",
    "FLOW_IAT_MIN",
    "FWD_IAT_TOTAL",
    "FWD_IAT_MEAN",
    "FWD_IAT_STD",
    "FWD_IAT_MAX",
    "FWD_IAT_MIN",
    "BWD_IAT_TOTAL",
    "BWD_IAT_MEAN",
    "BWD_IAT_STD",
    "BWD_IAT_MAX",
    "BWD_IAT_MIN",
    "FWD_PSH_FLAGS",
    "BWD_PSH_FLAGS",
    "FWD_URG_FLAGS",
    "BWD_URG_FLAGS",
    "FWD_HEADER_LENGTH",
    "BWD_HEADER_LENGTH",
    "FWD_PACKETS_S",
    "BWD_PACKETS_S",
    "MIN_PACKET_LENGTH",
    "MAX_PACKET_LENGTH",
    "PACKET_LENGTH_MEAN",
    "PACKET_LENGTH_STD",
    "PACKET_LENGTH_VARIANCE",
    "FIN_FLAG_COUNT",
    "SYN_FLAG_COUNT",
    "RST_FLAG_COUNT",
    "PSH_FLAG_COUNT",
    "ACK_FLAG_COUNT",
    "URG_FLAG_COUNT",
    "CWE_FLAG_COUNT",
    "ECE_FLAG_COUNT",
    "DOWN_UP_RATIO",
    "AVERAGE_PACKET_SIZE",
    "AVG_FWD_SEGMENT_SIZE",
    "AVG_BWD_SEGMENT_SIZE",
    "FWD_HEADER_LENGTH_1",
    "FWD_AVG_BYTES_BULK",
    "FWD_AVG_PACKETS_BULK",
    "FWD_AVG_BULK_RATE",
    "BWD_AVG_BYTES_BULK",
    "BWD_AVG_PACKETS_BULK",
    "BWD_AVG_BULK_RATE",
    "SUBFLOW_FWD_PACKETS",
    "SUBFLOW_FWD_BYTES",
    "SUBFLOW_BWD_PACKETS",
    "SUBFLOW_BWD_BYTES",
    "INIT_WIN_BYTES_FORWARD",
    "INIT_WIN_BYTES_BACKWARD",
    "ACT_DATA_PKT_FWD",
    "MIN_SEG_SIZE_FORWARD",
    "ACTIVE_MEAN",
    "ACTIVE_STD",
    "ACTIVE_MAX",
    "ACTIVE_MIN",
    "IDLE_MEAN",
    "IDLE_STD",
    "IDLE_MAX",
    "IDLE_MIN",
    "LABEL",
]

df_portscan.columns = col_portscan

db_con = conn_DB()
print(db_con)
  
if __name__=="__main__":
    # DB 초기화 : 테이블 생성
    Base.metadata.create_all(db_con)
    
    # Base.metadata.bind(db_con)
    # asyncio.run(migrate_table())
    
    # 데이터 입력
    # table_name = 'CSCIDS2017_BALANCED_ATTK'
    # if_exists = 'append' # 'replace', 'fail'
    # with db_con.connect() as con:
    #     df_balance.to_sql(
    #     name=table_name.upper(),
    #     con=con,
    #     if_exists=if_exists
    #     )

    table_name = 'CSCIDS2017_FRI_PM_PORTSCAN'
    if_exists = 'append' # 'replace', 'fail'
    with db_con.connect() as con:
        df_portscan.to_sql(
        name=table_name.upper(),
        con=con,
        if_exists=if_exists
        )

    # 입력 확인
    # session = Session(db_con)
    # stmt = select(CSCIDS2017_BALANCED_ATTK).where(CSCIDS2017_BALANCED_ATTK.FLOW_ID.in_([62015,41742]))
    # for Item in session.scalars(stmt):
    #     print(Item)