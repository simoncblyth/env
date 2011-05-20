from sa import SA, SABase
from dcs import DcsTableName as DTN

class DcsBase(SABase):
    """
    Base for mapped classes that have `id` and `date_time` attributes
    """
    def __repr__(self):
        return "%s %s %s " % ( self.__class__.__name__, self.id, self.date_time )


class DCS(SA):
    def __init__(self, dbconf ):
        """
        SQLAlchemy connection to database, performing
        table reflection and mappings from tables 
        
        Specializations:

        #. establishes standard query ordering, assuming a date_time attribute in tables

        TODO:

        #. get mapping to joins to work without duplicate attribute problems 

        """
        SA.__init__( self, dbconf )

    def subbase(self, dtn):
        """subclass to use, that can be dependent on table coordinate """
        return DcsBase

    def q(self, kls):
        return self.session.query(kls)
    def qa(self, kls):
        return self.session.query(kls).order_by(kls.date_time)
    def qd(self, kls):
        return self.session.query(kls).order_by(kls.date_time.desc())


def test_filter():

    from datetime import datetime
    cut = datetime( 2011,5,19, 9 )
    dtns = []
    for site in "DBNS".split():
        for det in "AD1 AD2".split():
            for qty in "HV HV_Pw".split():
                dtns.append( DTN(site, det, qty) )

    for dtn in dtns: 
        print "%" * 20, dtn, "%" * 20

        kls = dcs.kls(dtn)
        assert kls.db  == dcs
        assert kls.xtn == dtn

        print "first %r " % kls.first()
        print "last  %r " % kls.last()
        print "count %r " % kls.count()

        qa = dcs.qa(kls)
        qd = dcs.qd(kls)

        print "date_time ascending %s" % kls.__name__
        for i in qa.all():
            print i 
        print "date_time descending %s" % kls.__name__
        for i in qd.all():
            print i 
        print "before %s " % cut    ## hmmm a qbefore / qafter would avoid having to spill the kls
        for i in qa.filter(kls.date_time < cut ).all():
            print i
        print "after %s " % cut 
        for i in qa.filter(kls.date_time > cut ).all():
            print i


if __name__ == '__main__':

    dcs = DCS("dcs")
    print "mapped tables ... initially will be none, as is lazy"
    for t in dcs.meta.tables:
        print t 

    print "dynamic classes from DTN coordinates "
    for det in "AD1 AD2".split():
        for qty in "HV HV_Pw".split(): 
            dtn = DTN("DBNS", det , qty )
            kls = dcs.kls(dtn)
            assert str(kls.xtn) == str(dtn), "dtn %s xtn %s" % (dtn, kls.xtn)
            assert kls.db  == dcs
            print kls, kls.__name__

    print "mapped tables ... should be some now"
    for t in dcs.meta.tables:
        print t 

    print "autojoin"
    dtn = DTN("DBNS","AD1","HV:HV_Pw:id:P_")
    kls = dcs.kls(dtn)
    print kls, kls.last(), dir(kls)
 

