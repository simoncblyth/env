from sa import SA
from dcs import DcsTableName, Hv, Pw
from sqlalchemy.orm import mapper
# from sqlalchemy.sql import join

class DCS(SA):
    def __init__(self, dbconf ):
        """
        SQLAlchemy connection to database, performing
        table reflection and mappings from tables to classes 
        
        Specializations:

        #. selects tables of interest to reflect/map to classes
        #. establishes standard query ordering 

        TODO:

        #. looping over tables, table(class) generalization   
        #. mapped classes for all tables ?  
        #. class conjuring with auto naming ?

        #. get mapping to joins working (no FK so must specify onclause) without
           duplicate attribute problems ::
        
           hp_j = join( hv_t , pw_t , hv_t.c.id == pw_t.c.id )   

        """
        dtn = DcsTableName("DBNS", "AD2")     
        SA.__init__( self, dbconf , tables=(dtn.hv,dtn.pw,) )

        hv_t = self(dtn.hv)   
        pw_t = self(dtn.pw)

        mapper( Hv ,  hv_t ) 
        mapper( Pw ,  pw_t ) 
    
    def qa(self, cls):
        """query date_time ascending""" 
        return self.session.query(cls).order_by(cls.date_time)

    def qd(self, cls):
        """query date_time ascending""" 
        return self.session.query(cls).order_by(cls.date_time.desc())
  

if __name__ == '__main__':

    dcs = DCS("dcs")

    qa = dcs.qa(Hv)
    qd = dcs.qd(Hv)

    print "mapped tables"
    for t in dcs.meta.tables:
        print t 

    print "date_time ascending Hv"
    for i in qa.all():
        print i 

    print "date_time descending Hv"
    for i in qd.all():
        print i 

    from datetime import datetime
    cut = datetime( 2011,5,19, 9 )

    print "all"
    for i in qa.all():
       print i
    print "before %s " % cut 
    for i in qa.filter(Hv.date_time < cut ).all():
       print i
    print "after %s " % cut 
    for i in qa.filter(Hv.date_time > cut ).all():
       print i


