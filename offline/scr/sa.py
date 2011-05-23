import os
from sqlalchemy import Table, MetaData, create_engine
from sqlalchemy.orm import sessionmaker, mapper
from sqlalchemy.sql import join
from sqlalchemy.orm.attributes import InstrumentedAttribute

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


class SABase(object):
    """
    Shortcut classmethods on the dynamic classes
    """
    last  = classmethod(lambda kls:kls.db.qd(kls).first())
    first = classmethod(lambda kls:kls.db.qa(kls).first())
    count = classmethod(lambda kls:kls.db.q(kls).count())
    q = classmethod(lambda kls:kls.db.q(kls))
    qa = classmethod(lambda kls:kls.db.qa(kls))
    qd = classmethod(lambda kls:kls.db.qd(kls))
    qafter  = classmethod(lambda kls,cut:kls.db.qafter(kls,cut))
    qbefore = classmethod(lambda kls,cut:kls.db.qbefore(kls,cut))
    attribs = classmethod(lambda kls:filter(lambda k:isinstance(getattr(kls,k), InstrumentedAttribute ),dir(kls))) 
    asdict = property(lambda self:dict(zip(self.attribs(),map(lambda k:getattr(self,k),self.attribs()))))

    def delta_dict(self, other):
        """
        Generic delta-ing 
        """
        dd = {}
        for k in self.attribs():
             vself  = getattr( self, k  )
             vother = getattr( other, k )
             dd[k] = vself - vother           
        return dd

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

    def _kls(self, xtn):
        """
        Dynamic creation of class to represent table row instances 
        with potentially table dependant base class
        """
        kln = xtn.kln                   
        Base = self.subbase( xtn )    
        kls = type(kln,(Base,),dict(db=self,xtn=xtn))    
        return kls

    def _map_properties(self, j, tb ):
        """
        :j param: the join instance
        :tb param: tiebreaker prefix

        when mapping to a join, there is tendency for property name collisions
        in the mapped class, avoid collisions by using the key name which is
        prefixed by table name 
        """ 
        props = {}
        for k in j.c.keys():
            v = j.c.get(k)      # Column instance
            n = v.name
            if n in props:
               props["%s%s"% (tb,n)] = v
            else:
               props[n] = v 
        return props

    def _mapclass(self, xtn):
        """
        map single table or join to a dynamic class
        """
        print "mapclass for xtn %s " % xtn
        kls = self._kls(xtn)
        if xtn.isjoin:
            j,tb = self._join( *xtn.jbits() ) 
            mapper( kls , j , properties=self._map_properties(j,tb) )
        else:
            tab = self.table( str(xtn))
            mapper( kls , tab )
        self.classes[kls.__name__] = kls  

    def kls(self, xtn ):
        """Return mapped dynamic class from a xtn instance"""
        kln = xtn.kln                 # dynamic class name
        if kln not in self.classes:
            self._mapclass(xtn)
        return self.classes[kln]

    def _join(self, tna, tnb, ja, tb ):
        """
        Returns the join of the 2 named tables

        :tna param: coordinates of first table
        :tnb param: coordinates of seconf table 
        :ja param:  where clause join attribute, eg "id" or "SEQNO"
        :tb param:  tiebreaker for name collisions

        """ 
        a = self.table(str(tna))
        b = self.table(str(tnb))
        return join( a , b , getattr(a.c,ja) == getattr(b.c, ja) ), tb

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
        tn = str(tn)
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

