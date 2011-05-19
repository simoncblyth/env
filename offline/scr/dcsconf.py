from sa import SA
from dcs import DcsTableName as DTN, DcsBase
from sqlalchemy.orm import mapper

class DCS(SA):
    def __init__(self, dbconf ):
        """
        SQLAlchemy connection to database, performing
        table reflection and mappings from tables 
        
        Specializations:

        #. selects tables of interest to reflect/map to classes
        #. establishes standard query ordering 
        #. dynamically generates classes to map to  

        TODO:

        #. get mapping to joins to work without duplicate attribute problems 
           (no FK so must specify onclause) ::
        
           from sqlalchemy.sql import join
           hp_j = join( hv_t , pw_t , hv_t.c.id == pw_t.c.id )   

        """
        instances = DTN.instances()
        tables = map( str, instances )
        SA.__init__( self, dbconf , tables=tables )

        self.classes = {}

        for dtn in instances:
            tn = str(dtn)                           # table name
            gcln = dtn.gcln                         # dynamic class name
            gcls = type( gcln, (DcsBase,),{})       # dynamic class creation 

            self.classes[gcln] = gcls               # record dynamic classes, keyed by name
            mapper( gcls , self.meta.tables[tn] )   # map dynamic class to reflected table 
    
    def qa(self, cls):
        """query date_time ascending""" 
        return self.session.query(cls).order_by(cls.date_time)

    def qd(self, cls):
        """query date_time ascending""" 
        return self.session.query(cls).order_by(cls.date_time.desc())
  

if __name__ == '__main__':

    from datetime import datetime
    cut = datetime( 2011,5,19, 9 )

    dcs = DCS("dcs")
    print "mapped tables"
    for t in dcs.meta.tables:
        print t 

    ## over dynamic classes
    for gcln, gcls in dcs.classes.items(): 
        qa = dcs.qa(gcls)
        qd = dcs.qd(gcls)

        print "date_time ascending %s" % gcln
        for i in qa.all():
            print i 
        print "date_time descending %s" % gcln
        for i in qd.all():
            print i 
        print "before %s " % cut 
        for i in qa.filter(gcls.date_time < cut ).all():
            print i
        print "after %s " % cut 
        for i in qa.filter(gcls.date_time > cut ).all():
            print i


