from sa import SA, SABase

class DcsBase(SABase):
    """Base for mapped classes that have `id` and `date_time` attributes """
    def __repr__(self):
        return "%s %s %s " % ( self.__class__.__name__, self.id, self.date_time )

    def contextrange(self, interval):
        """
        Construct target context range for this source instance, translating the source date_time into target 
        timeStart/timeEnd using the interval argument
        """  
        return dict(timeStart=self.date_time,timeEnd=self.date_time+interval,siteMask=self.xtn.sitemask,subsite=self.xtn.subsite)

    def age(self, prev):
        return self.date_time - prev.date_time


class DCS(SA):
    def __init__(self, dbconf ):
        """
        SQLAlchemy connection to database, performing
        table reflection and mappings from tables 
        
        Specializations:
        #. standard query ordering, assuming a date_time attribute in tables

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
    def qafter(self, kls, cut ):
        return self.session.query(kls).order_by(kls.date_time).filter(kls.date_time >= cut)
    def qbefore(self, kls, cut ):
        return self.session.query(kls).order_by(kls.date_time).filter(kls.date_time < cut)
         

if __name__ == '__main__':
    pass


