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
    def __init__(self, dbconf , tables=[] ):
        meta = MetaData()
        engine = create_engine( sa_url(dbconf), echo=False )

        if len(tables) == 0:
            meta.reflect(bind=engine)    
        else:
            for t in tables:
                tt = Table( t , meta, autoload=True, autoload_with=engine)

        Session.configure(bind=engine)
        session = Session()

        self.meta = meta
        self.session = session

    def add(self, obj):
        self.session.add( obj )

    def commit(self):
        self.session.commit()  

    def __call__(self, tn ):
        return self.meta.tables[tn]

if __name__ == '__main__':
    dcs = SA("dcs")
    for t in dcs.meta.tables:
        print t

