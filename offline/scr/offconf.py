
from sa import SA, SABase
from off import OffTableName as OTN, instances 
from sqlalchemy.orm import mapper

class OffBasePay(SABase):
    """
    Base for mapped payload classes
    """
    def __repr__(self):
        return "OBP %s %s %s" % ( self.__class__.__name__, self.SEQNO, self.ROW_COUNTER  )

class OffBaseVld(SABase):
    """
    Base for mapped validity classes
    """
    def __repr__(self):
        return "OBV %s %s %s %s %s %s" % ( self.__class__.__name__, self.SEQNO, self.TIMESTART, self.TIMEEND, self.VERSIONDATE, self.INSERTDATE )

class OffBasePayVld(SABase):
    """
    Base for mapped validity classes
    """
    def __repr__(self):
        return "OBPV %s %s %s %s %s %s %s" % ( self.__class__.__name__, self.SEQNO, self.ROW_COUNTER, self.TIMESTART, self.TIMEEND, self.VERSIONDATE, self.INSERTDATE )



class OFF(SA):
    basemap = dict(Pay=OffBasePay, Vld=OffBaseVld)
    def __init__(self, dbconf ):
        """
        SQLAlchemy connection to database, performing
        table reflection and mappings from tables to classes 
        
        Specializations:
 
        #. establishes standard query ordering 

        """
        SA.__init__( self, dbconf )

    def subbase(self, otn):
        """subclass to use, that can be dependent on table coordinate"""
        if otn.flv in self.basemap:
            return self.basemap[otn.flv]
        else:
            return OffBasePayVld

    def q(self, kls):
        return self.session.query(kls)
    def qa(self, kls):
        return self.session.query(kls).order_by(kls.SEQNO)
    def qd(self, kls):
        return self.session.query(kls).order_by(kls.SEQNO.desc())
    def qafter(self, kls, cut ):
        return self.session.query(kls).order_by(kls.SEQNO).filter(kls.SEQNO >= cut)
    def qbefore(self, kls, cut ):
        return self.session.query(kls).order_by(kls.SEQNO).filter(kls.SEQNO < cut)
                   


def test_filter():
    for otn in instances(): 
        kls = off.kls( otn )
        assert kls.xtn == otn    ## kls knows where it came from
        assert kls.db  == off 

        print "first %r " % kls.first()
        print "last  %r " % kls.last()
        print "count %r " % kls.count()

        print "%" * 20, otn, "%" * 20

        #qa = off.qa(kls)
        #qd = off.qd(kls)
        qa = kls.qa()    ## using classmethod shortcut
        qd = kls.qd()

        la = qa.first()   
        print "first in SEQNO ascending order, ie 1st SEQNO", la, la.SEQNO
        ld = qd.first()   
        print "first in SEQNO descending order, ie last SEQNO", ld, ld.SEQNO
        na = qa.count()
        nd = qd.count()
        assert na == nd
        print "qa count %d qd count %d " % ( na, nd )

        if otn.flv == "Vld":
            cut = ld.SEQNO - 3
            print "SEQNO after %s " % cut 
            for i in qa.filter(kls.SEQNO > cut ).all():
                print i


if __name__ == '__main__':
    pass
    off = OFF("recovered_offline_db")

    #test_filter()

    print "lookup dynamic classes from OTN coordinates "
    for qty in "PmtHv AdTemp".split():
        kls = off.kls(OTN("Dcs",qty,"Vld"))
        print kls, kls.__name__

    print "manual join"
    j = off._join("DcsPmtHv","DcsPmtHvVld", "SEQNO", "V_")
    print "j", j

    print "auto join and map" 
    otn = OTN("Dcs","AdTemp","Pay:Vld:SEQNO:V_") 
    kls = off.kls(otn)
    print kls

    last = kls.last()

    print "last", last, dir(last)
    



