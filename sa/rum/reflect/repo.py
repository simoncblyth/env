"""
   http://docs.python-rum.org/tip/user/customize/repository.html

Repositories can be overrided for particular models to alter the way 
these are saved, updated, retrieved, etc. in a similar way as the above. 
For example, say we?re using RumAlchemy to handle our SQLAlchemy mapped 
classes and we want to override how Engine instances are created so we can set a timestamp field:

"""
import datetime
from mymodel import Dbi
from rum.repository import RepositoryFactory
from rumalchemy import SARepository

class DbiRepository(SARepository):
    def create(self, data):
        # Delegate the call to the superclass
        obj = super(DbiRepository, self).create(data)
        obj.created_on = datetime.datetime.now()
        return obj

RepositoryFactory.register(DbiRepository, Dbi)



