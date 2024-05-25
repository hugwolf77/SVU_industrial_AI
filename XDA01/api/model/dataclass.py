import uuid
from datetime import datetime
from pydantic import BaseModel, Field

""" model Input Data specify """

class DataInput(BaseModel):
    # ReqTime : str = Field(min_length=4, max_length=10)
    Index : list[int] 
    date : list[datetime] 
    HUFL : list[float] 
    HULL : list[float] 
    MUFL : list[float] 
    MULL : list[float] 
    LUFL : list[float] 
    LULL : list[float] 
    OT :   list[float] 

    class Config:
        orm_mode = True

class PredictOutput(BaseModel):
    date : list[str] = Field(str)
    HUFL : list[float]    = Field(float)
    HULL : list[float]    = Field(float)
    MUFL : list[float]    = Field(float)
    MULL : list[float]    = Field(float)
    LUFL : list[float]    = Field(float)
    LULL : list[float]    = Field(float)
    OT   : list[float]    = Field(float)

    class Config:
        orm_mode = True