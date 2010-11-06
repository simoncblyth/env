
from sqlalchemy import Column, Integer, String, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql.expression import ClauseElement

Base = declarative_base()

class Qry(Base): 
    __tablename__ = "Qry"
    id = Column(Integer, primary_key=True)
    name = Column(String(30)) 
    def __init__(self, name ):
        self.name = name 

    def __repr__(self):
        return "<Qry %s>" % self.name





def get_or_create(session, model, defaults={}, **kwargs):
    instance = session.query(model).filter_by(**kwargs).first()
    if instance:
        return instance, False
    else:
        params = dict((k, v) for k, v in kwargs.iteritems() if not isinstance(v, ClauseElement))
        print params

        params.update(defaults)
        instance = model(**params)
        session.add(instance)
        return instance, True








__all__ = "Base Qry".split()


