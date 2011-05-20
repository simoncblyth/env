from sa import SA
from dcs import DcsTableName as DTN

class DcsBase(object):
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

        #. selects tables of interest to reflect/map to classes
        #. establishes standard query ordering, assuming a date_time attribute in tables
        #. dynamically generates classes to map to  

        TODO:

        #. get mapping to joins to work without duplicate attribute problems 
           (no FK so must specify onclause) ::
        
           from sqlalchemy.sql import join
           hp_j = join( hv_t , pw_t , hv_t.c.id == pw_t.c.id )   

        """
        SA.__init__( self, dbconf )

    def subbase(self, dtn):
        """
        subclass to use, that can be dependent on table coordinate 
        """
        return DcsBase

    ## the below specialization boils down to : date_time
    def q_(self,dtn):
	kls = self.kls(dtn)
	return self.session.query(kls)
    def qa_(self,dtn):
        """query date_time ascending"""
        kls = self.kls(dtn)
	return self.session.query(kls).order_by(kls.date_time)
    def qd_(self,dtn):
        """query date_time descending"""
        kls = self.kls(dtn) 
        return self.session.query(kls).order_by(kls.date_time.desc())


    def q(self, kls):
        return self.session.query(kls)
    def qa(self, kls):
        return self.session.query(kls).order_by(kls.date_time)
    def qd(self, kls):
        return self.session.query(kls).order_by(kls.date_time.desc())





if __name__ == '__main__':

    from datetime import datetime
    cut = datetime( 2011,5,19, 9 )

    dcs = DCS("dcs")
    print "mapped tables ... initially will be none, as is lazy"
    for t in dcs.meta.tables:
        print t 

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

    print "lookup dynamic classes from DTN coordinates "

    for det in "AD1 AD2".split():
        dtn = DTN("DBNS", det , "HV")
        kls = dcs.kls(dtn)
        assert str(kls.xtn) == str(dtn), "dtn %s xtn %s" % (dtn, kls.xtn)
        assert kls.db  == dcs
        print kls, kls.__name__

        drn = DTN("DBNS", det , "HV_Pw")
        kls = dcs.kls(dtn)
        assert str(kls.xtn) == str(dtn), "dtn %s xtn %s" % (dtn, kls.xtn)   
        assert kls.db  == dcs
        print kls, kls.__name__   

    print "mapped tables ... should be some now"
    for t in dcs.meta.tables:
        print t 




