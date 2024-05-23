import uuid
from pydantic import BaseModel, Field

""" model Input Data specify """

class DataInput(BaseModel):
    NM : str = Field(min_length=4, max_length=10)
    x : list[float] = Field()


class PredictOutput(BaseModel):
    prediction : int