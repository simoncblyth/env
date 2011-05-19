
from sa import SA
from off import OffTableName, Vld
from sqlalchemy.orm import mapper
        
otn = OffTableName()

class OFF(SA):
    def __init__(self, dbconf ):
        """
        SQLAlchemy connection to database, performing
        table reflection and mappings from tables to classes 
        
        Specializations:
 
        #. selects tables of interest to reflect upon  
        #. maps some of these tables to classes  
        #. establishes standard query ordering 

        """
        SA.__init__( self, dbconf, tables=otn.dbi_pairs(["hv"])  )
        hvv_t = self(otn.hv+"Vld")    # sqlalchemy.schema.Table objects
        mapper( Vld , hvv_t )

    def qa(self, cls):
        """query SEQNO ascending""" 
        return self.session.query(cls).order_by(cls.SEQNO)

    def qd(self, cls):
        """query SEQNO descending""" 
        return self.session.query(cls).order_by(cls.SEQNO.desc())


if __name__ == '__main__':
    pass
    off = OFF("recovered_offline_db")
    for t in off.meta.tables:
        print t

    q = off.qd(Vld)
    print q.count()
   
    last = q.first()   ## first in SEQNO descending order, ie the last SEQNO   
    print last
    print last.SEQNO
 

 
