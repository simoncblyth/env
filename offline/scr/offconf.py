
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
        SA.__init__( self, dbconf )

    def subbase(self, otn):
        """
        subclass to use, that can be dependent on table coordinate 
        """
        return OffBasePay if otn.flv == "Pay" else OffBaseVld


    def q_(self, otn):
        kls = self.kls(otn)
        return self.session.query(kls)
    def qa_(self, otn):
        """query SEQNO ascending""" 
        kls = self.kls(otn)
        return self.session.query(kls).order_by(kls.SEQNO)
    def qd_(self, otn):
        """query SEQNO descending""" 
        kls = self.kls(otn)
        return self.session.query(kls).order_by(kls.SEQNO.desc())


    def q(self, kls):
        return self.session.query(kls)
    def qa(self, kls):
        return self.session.query(kls).order_by(kls.SEQNO)
    def qd(self, kls):
        return self.session.query(kls).order_by(kls.SEQNO.desc())
                  


if __name__ == '__main__':
    pass
    off = OFF("recovered_offline_db")

    for otn in instances(): 
        kls = off.kls( otn )

        assert kls.xtn == otn    ## kls knows where it came from
        assert kls.db  == off 

        print "%" * 20, otn, "%" * 20

        qa = off.qa(kls)
        qd = off.qd(kls)

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



 
