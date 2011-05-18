import os
from sqlalchemy import MetaData, create_engine

def sa_url( sect , path="~/.my.cnf" ):
    """
    Provide SQLAlchemy URL for the config file section `sect`
    """ 
    from ConfigParser import ConfigParser
    cfp = ConfigParser()
    cfp.read( map(lambda _:os.path.expanduser(_), [path] ))
    cfg = dict(cfp.items(sect))
    return "mysql://%(user)s:%(password)s@%(host)s/%(database)s" % cfg

def sa_meta( url , **kwa ):
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
    m = MetaData()
    e = create_engine( url, **kwa )
    m.reflect(bind=e)       
    return m



