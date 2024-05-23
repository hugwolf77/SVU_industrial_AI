
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, Float, String, Integer

class Base(DeclarativeBase):
    """Base database model"""
    pass

class dataTB(Base):
    __tablename__ = "IOT_SENSOR"
    Index = Column(Integer, primary_key=True)
    pass

