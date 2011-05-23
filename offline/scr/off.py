
class OffTableName(object):
    """
    Supply Offline DB table names, given some ctor arguments
    This class is intended as insulation against table name changes.
    
    Note, old scraper code features "DcsAdPmtHv" which does not appear 
    in offline_db 
    """ 

    kln = property(lambda self:'G%s' % ( str(self) ))      #  rationalized class name G<tablename> 

    pfxList = "Dcs".split()
    qtyList = "PmtHv AdTemp".split()
    flvList = "Pay Vld".split()
 
    def __init__(self, pfx, qty, flv):
        self.pfx = pfx
        self.qty = qty
        self.flv = flv
        assert pfx in self.pfxList, pfx 
        assert qty in self.qtyList, qty 
        assert flv in self.flvList or self.isjoin, flv 

    isjoin = property(lambda self:self.flv.find(":") > -1)
    def jbits(self):
       assert self.isjoin
       a,b,j,tb = self.flv.split(":")               ## tb is tiebreaker prefix for name collisions 
       xa = OffTableName( self.pfx, self.qty, a )
       xb = OffTableName( self.pfx, self.qty, b )
       return xa,xb,j,tb

    def __repr__(self):
        return "OTN %-15s %-10s %-10s %-10s     %-10s" % ( str(self), self.pfx, self.qty, self.flv, self.kln )

    def __str__(self):
         if self.flv == "Pay":
             flv = ""
         else:
             flv = self.flv
         return "%s%s%s" % (self.pfx, self.qty, flv ) 


if __name__ == '__main__':
    pass
