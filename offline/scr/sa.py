import os
from sqlalchemy import Table, MetaData, create_engine
from sqlalchemy.orm import sessionmaker, mapper
Session = sessionmaker()


def sa_url( sect , path="~/.my.cnf" ):
    """
    Provide SQLAlchemy URL for the config file section `sect`
    """ 
    from ConfigParser import ConfigParser
    cfp = ConfigParser()
    cfp.read( map(lambda _:os.path.expanduser(_), [path] ))
    cfg = dict(cfp.items(sect))
    return "mysql://%(user)s:%(password)s@%(host)s/%(database)s" % cfg

class SA(object):
    """
    Reflect schema of all tables in database at `url`

    How to handle connections to multiple DB ? 

    Paraphrasing 
        http://www.sqlalchemy.org/docs/07/core/schema.html?highlight=multiple%20metadata#binding-metadata-to-an-engine-or-connection

    Application has multiple schemas that correspond to different engines. 
    Using one MetaData for each schema, bound to each engine, provides a decent place to delineate between the schemas. 
    The ORM will also integrate with this approach, where the Session will naturally use the engine that is bound to each 
    table via its metadata (provided the Session itself has no bind configured.).

    Adopt simple approach of binding the engine to the metadata

    """  
    def __init__(self, dbconf ):
        meta = MetaData()
        engine = create_engine( sa_url(dbconf), echo=False )
        Session.configure(bind=engine)
        session = Session()
        
        self.engine = engine
        self.session = session
        self.meta = meta
        self.classes = {}

    def _mapclass(self, xtn):
        print "mapclass for xtn %s " % xtn
        tn = str(xtn)
        tab = self.table(tn)           # will reflect the table if not already done
        kln = xtn.kln                  # dynamic class name 
        Base = self.subbase( xtn )     # potentially table dependant base class
        kls = type(kln,(Base,),dict(db=self,xtn=xtn))     # dynamic subclass creation
        mapper( kls , tab )
        self.classes[kln] = kls        # ... hmmm maybe key by xtn ?

    def kls(self, xtn ):
        """Return mapped dynamic class from a xtn instance"""
        kln = xtn.kln                 # dynamic class name
        if kln not in self.classes:
            self._mapclass(xtn)
        return self.classes[kln]

    def reflect(self, tn ):
        """
        Reflect on the table, recording it in the meta
        """
        tt = Table( tn , self.meta,  autoload=True, autoload_with=self.engine)
        assert self.meta.tables[tn] == tt

    def table(self, tn ):
        """
        Return the sqlalchemy.schema.Table representation of a table, reflect upon
        the table if not already done 
        """
        if tn not in self.meta.tables:
            self.reflect(tn)
        return self.meta.tables[tn]

    def add(self, obj):
        self.session.add( obj )

    def commit(self):
        self.session.commit()  

    #def __call__(self, tn ):
    #    return self.meta.tables[tn]

if __name__ == '__main__':
    dcs = SA("dcs")
    for t in dcs.meta.tables:
        print t

