from sqlalchemy.orm import Session
from DB.DBmodel.dataTB import Base, dataTB
from sqlalchemy import select

def get_items(db: Session):
    return db.query(dataTB).all()

def get_item(ind,db:Session):
    return db.query(dataTB).filter(dataTB.Index.in_([ind])).first()

