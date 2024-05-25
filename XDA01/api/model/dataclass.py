import uuid
from typing import List
from datetime import datetime
from pydantic import BaseModel, Field

""" model Input Data specify """

class DataInput(BaseModel):
    # ReqTime : str = Field(min_length=4, max_length=10)
    Index : List[int] = Field()
    date : List[datetime] = Field() 
    HUFL : List[float] = Field()
    HULL : List[float] = Field()
    MUFL : List[float] = Field()
    MULL : List[float] = Field()
    LUFL : List[float] = Field()
    LULL : List[float] = Field()
    OT : List[float] = Field()

    class Config:
        orm_mode = True

class PredictOutput(BaseModel):
    prediction : List[float] = Field()

    class Config:
        orm_mode = True