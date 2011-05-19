"""
# no FK so must specify the onclause
# from sqlalchemy.sql import join
# hp_j = join( hv_t , pw_t , hv_t.c.id == pw_t.c.id )   ## subsequent mappings give duplicate attribute problems    

"""

from sa import SA
from dcstable import DcsTableName, Hv
from dcsfake import LCRFaker

from sqlalchemy.orm import mapper
from datetime import datetime


if __name__ == '__main__':

    # table name provider
    dtn = DcsTableName("DBNS", "AD2")     

    # SQLAlchemy connection to database and table reflection
    dcs = SA("dcs", tables=(dtn.hv,dtn.pw,) )


    off = SA("off")  ## all tables


    # sqlalchemy.schema.Table objects
    hv_t = dcs(dtn.hv)
    pw_t = dcs(dtn.pw)

    # maps hv table to Hv class
    mapper( Hv ,  hv_t ) 
    
    # instance of mapped class   
    hv = Hv()    

    # fake the LCR attributes
    fk = LCRFaker()
    fk(hv)

    hv.id = 3
    hv.date_time = datetime.now()

    dcs.add(hv)   
    dcs.commit()

    # access instances with filtering 
    cut = datetime( 2011,5,19, 9 )

    for i in dcs.session.query(Hv).all():
       print i.id, i.date_time

    for i in dcs.session.query(Hv).filter(Hv.date_time < cut ).all():
       print i.id, i.date_time

    for i in dcs.session.query(Hv).filter(Hv.date_time > cut ).all():
       print i.id, i.date_time



