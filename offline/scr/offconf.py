from sa import SA, SABase

class OffBasePay(SABase):
    """Base for mapped payload classes"""
    def __repr__(self):
        return "OBP %s %s %s" % ( self.__class__.__name__, self.SEQNO, self.ROW_COUNTER  )

class OffBaseVld(SABase):
    """Base for mapped validity classes"""
    def __repr__(self):
        return "OBV %s %s %s %s %s %s" % ( self.__class__.__name__, self.SEQNO, self.TIMESTART, self.TIMEEND, self.VERSIONDATE, self.INSERTDATE )

class OffBasePayVld(SABase):
    """Base for mapped validity classes"""
    def __repr__(self):
        return "OBPV %s %s %s %s %s %s %s" % ( self.__class__.__name__, self.SEQNO, self.ROW_COUNTER, self.TIMESTART, self.TIMEEND, self.VERSIONDATE, self.INSERTDATE )

class OFF(SA):
    basemap = dict(Pay=OffBasePay, Vld=OffBaseVld)
    def __init__(self, dbconf ):
        """
        SQLAlchemy connection to database, performing
        table reflection and mappings from tables to classes 
        
        Specializations:
 
        #. standard query ordering based on SEQNO
        #. table dependant base class for row/join mapped class instances  

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
                   

if __name__ == '__main__':
    pass

