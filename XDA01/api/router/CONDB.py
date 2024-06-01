import logging
import uuid
from typing import Optional
from fastapi import APIRouter, Depends,  HTTPException, status

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from API_app.apis import test # main logic
from API_app.model import dataclass

from DB.DBmodel.dataTB import Base
from DB.DBmodel import CSCIDS2017
from DB.async_connection import get_db_session


logger = logging.getLogger("conDB")
conDB = APIRouter(prefix="/conDB",) 


@conDB.get("/read", status_code=status.HTTP_200_OK)
async def read_Item(db:AsyncSession=Depends(get_db_session)) -> list[CSCIDS2017.CSCIDS2017_balanced_attk]:
    Items = await db.scalars(select(CSCIDS2017.CSCIDS2017_balanced_attk))
    return [dataclass.CSCIDS_balanced.model_validate(Item) for Item in  Items]

@conDB.get("/readAll", status_code=status.HTTP_200_OK)
async def read_Item(flowid, db:AsyncSession=Depends(get_db_session)) -> CSCIDS2017.CSCIDS2017_balanced_attk:
    Item = await db.get(CSCIDS2017.CSCIDS2017_balanced_attk, flowid)
    if Item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Item does not exist",
        )
    return dataclass.CSCIDS_balanced.model_validate(Item)




###
# @conDB.get("/")
# async def DB_index(db:AsyncSession=Depends(get_db_session)):
#     res = test.test_index(db=db)
#     logger.info("Client got test response {}, by use session :{}".format(res, db))
#     return {"res" : res,}
# @conDB.post("/post")
# def create_Item(Input:DataInput,db:Session=Depends(get_db_session)):
#     DB_Table = Base()
#     db.add(DB_Table)
#     db.commit()
#     return Input

# @conDB.put("/")
# def update_Item

# @conDB.delete("/")
# def delete_Item():
    