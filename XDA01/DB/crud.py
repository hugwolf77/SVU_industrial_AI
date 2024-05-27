from sqlalchemy.orm import Session
from DB.DBmodel.dataTB import Base, ETT_H_1, ETT_H_2, ETT_M_1, ETT_M_2
from sqlalchemy import select

def get_items(db: Session):
    return db.query(ETT_H_1).all()

def get_item(ind,db:Session):
    return db.query(ETT_H_1).filter(ETT_H_1.Index.in_([ind])).first()

# def get_ETT_DataReed(db_con, TB):
#     stmt = select(TB)
#     with Session(db_con) as ses:
#         data = ses.execute(stmt)
#     return data


