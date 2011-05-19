"""
# no FK so must specify the onclause
# from sqlalchemy.sql import join
# hp_j = join( hv_t , pw_t , hv_t.c.id == pw_t.c.id )   ## subsequent mappings give duplicate attribute problems    

"""

from sa import SA
from dcs import DcsTableName, Hv, Pw
from dcsfake import LCRFaker

from sqlalchemy.orm import mapper
from datetime import datetime


class Feed(object):
    """
    Feed instances of the mapped class specified, usage:

        src = Feed(Hv)
        for i in src:
            print i 

    where to put the time logic ?

    """ 
    def __init__(self, kls ):
        self.kls = kls

    def upto(self, cut):
        """
        _dcsDb.cursor.execute("SELECT * FROM " + self.info["hv"] + " WHERE date_time >= %s LIMIT 1", (self.info["nextTime"], ))
        """
        pass 

    def last(self, cut ):
        """
         _dcsDb.cursor.execute("SELECT date_time FROM " + self.info["hv"] + " WHERE date_time < %s ORDER BY date_time DESC LIMIT 1", (timeStart,))
        """
        pass


if __name__ == '__main__':

    # table name provider
    dtn = DcsTableName("DBNS", "AD2")     

    # SQLAlchemy connection to database and table reflection
    dcs = SA("dcs", tables=(dtn.hv,dtn.pw,) )

    # sqlalchemy.schema.Table objects
    hv_t = dcs(dtn.hv)
    pw_t = dcs(dtn.pw)

    # map tables to classes
    mapper( Hv ,  hv_t ) 
    mapper( Pw ,  pw_t ) 
    
    # create Hv instance, fake the LCR attributes and write to dcs
    if 0:
        hv = Hv()    
        fk = LCRFaker()
        fk(hv)
        hv.id = 3
        hv.date_time = datetime.now()
        dcs.add(hv)   
        dcs.commit()

    # access instances with filtering 
    cut = datetime( 2011,5,19, 9 )
    
    q = dcs.session.query(Hv).order_by(Hv.date_time)

    print "all"
    for i in q.all():
       print i
    print "before %s " % cut 
    for i in q.filter(Hv.date_time < cut ).all():
       print i
    print "after %s " % cut 
    for i in q.filter(Hv.date_time > cut ).all():
       print i


    #off = SA("recovered_offline_db")


