
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import Column, Float, String, Integer

class Base(DeclarativeBase):
    """Base database model"""
    pass

class ETT_H_1(Base):
    __tablename__ = "ETT_H_1"
    Index = Column(Integer, primary_key=True)
    date = Column(Float)
    HUFL = Column(Float)
    HULL = Column(Float)
    MUFL = Column(Float)
    MULL = Column(Float)
    LUFL = Column(Float)
    LULL = Column(Float)
    OT = Column(Float)

class ETT_H_2(Base):
    __tablename__ = "ETT_H_2"
    Index = Column(Integer, primary_key=True)
    date = Column(Float)
    HUFL = Column(Float)
    HULL = Column(Float)
    MUFL = Column(Float)
    MULL = Column(Float)
    LUFL = Column(Float)
    LULL = Column(Float)
    OT = Column(Float)

class ETT_M_1(Base):
    __tablename__ = "ETT_M_1"
    Index = Column(Integer, primary_key=True)
    date = Column(Float)
    HUFL = Column(Float)
    HULL = Column(Float)
    MUFL = Column(Float)
    MULL = Column(Float)
    LUFL = Column(Float)
    LULL = Column(Float)
    OT = Column(Float)

class ETT_M_2(Base):
    __tablename__ = "ETT_M_2"
    Index = Column(Integer, primary_key=True)
    date = Column(Float)
    HUFL = Column(Float)
    HULL = Column(Float)
    MUFL = Column(Float)
    MULL = Column(Float)
    LUFL = Column(Float)
    LULL = Column(Float)
    OT = Column(Float)