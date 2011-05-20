
from sa import SA
from off import OffTableName as OTN, instances 
from sqlalchemy.orm import mapper
        
class OffBasePay(object):
    """
    Base for mapped payload classes
    """
    def __repr__(self):
        return "%s %s %s" % ( self.__class__.__name__, self.SEQNO, self.ROW_COUNTER  )

class OffBaseVld(object):
    """
    Base for mapped validity classes
    """
    def __repr__(self):
        return "%s %s %s %s %s %s" % ( self.__class__.__name__, self.SEQNO, self.TIMESTART, self.TIMEEND, self.VERSIONDATE, self.INSERTDATE )


class OFF(SA):
    def __init__(self, dbconf ):
        """
        SQLAlchemy connection to database, performing
        table reflection and mappings from tables to classes 
        
        Specializations:
 
        #. selects tables of interest to reflect upon  
        #. maps some of these tables to classes  
        #. establishes standard query ordering 

        TODO:

        #. refactor to eliminate duplication between this and DCS

        """
        #instances = OTN.instances()
        #tables = map( str, instances )
        #SA.__init__( self, dbconf, tables=tables  )
        SA.__init__( self, dbconf )

        #self.classes = {}
        #for otn in instances:
        #    tn = str(otn)
        #    gcln = otn.gcln
        # flv dependent base class for the dynamic class  
        #   gcls = type( gcln, (bcls,), {})
        #    self.classes[gcln] = gcls               # record dynamic classes, keyed by name
        #    mapper( gcls , self.meta.tables[tn] )   # map dynamic class to reflected table 
        #    pass
        #self.instances = instances              

    def subbase(self, otn):
        """
        subclass to use, that can be dependent on table coordinate 
        """
        return OffBasePay if otn.flv == "Pay" else OffBaseVld

    def qa(self, otn):
        """query SEQNO ascending""" 
        kls = self.kls(otn)
        return self.session.query(kls).order_by(kls.SEQNO)

    def qd(self, otn):
        """query SEQNO descending""" 
        kls = self.kls(otn)
        return self.session.query(kls).order_by(kls.SEQNO.desc())


if __name__ == '__main__':
    pass
    off = OFF("recovered_offline_db")

    for otn in instances(): 
        kls = off.kls( otn )
        print "%" * 20, otn, "%" * 20

        qa = off.qa(otn)
        qd = off.qd(otn)

        la = qa.first()   
        print "first in SEQNO ascending order, ie 1st SEQNO", la, la.SEQNO

        ld = qd.first()   
        print "first in SEQNO descending order, ie last SEQNO", ld, ld.SEQNO

        na = qa.count()
        nd = qd.count()
        print "qa count %d qd count %d " % ( na, nd )

        if otn.flv == "Vld":
            cut = ld.SEQNO - 3
            print "SEQNO after %s " % cut 
            for i in qa.filter(kls.SEQNO > cut ).all():
                print i

    print "lookup dynamic classes from OTN coordinates "
    for qty in "PmtHv AdTemp".split():
        kls = off.kls(OTN("Dcs",qty,"Vld"))
        print kls, kls.__name__



 
