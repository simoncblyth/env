
class OffTableName(object):
    """
    Supply Offline DB table names, given some ctor arguments
    This class is intended as insulation against table name changes.
    
    Note, old scraper code features "DcsAdPmtHv" which does not appear 
    in offline_db 
    """ 

    def instances(cls):
        """
        Selection of OffTableName instances
        """
        otns = []
        for pfx in cls.pfxList:
            for qty in cls.qtyList:
                for flv in cls.flvList:
                    otns.append( cls(pfx,qty,flv) )
        return otns
    instances = classmethod(instances) 
    tables    = classmethod(lambda cls:map(str, cls.instances() ))
    classes   = classmethod(lambda cls:map(lambda _:_.gcln, cls.instances() ))

    gcln = property(lambda self:'G%s' % ( str(self) ))      #  rationalized class name G<tablename> 

    pfxList = "Dcs".split()
    qtyList = "PmtHv AdTemp".split()
    flvList = "Pay Vld".split()

    def __init__(self, pfx, qty, flv):
        assert pfx in self.pfxList, pfx 
        assert qty in self.qtyList, qty 
        assert flv in self.flvList, flv 

        self.pfx = pfx
        self.qty = qty
        self.flv = flv

    def __repr__(self):
        return "OTN %-15s %-10s %-10s %-10s     %-10s" % ( str(self), self.pfx, self.qty, self.flv, self.gcln )

    def __str__(self):
         if self.flv == "Pay":
             flv = ""
         else:
             flv = self.flv
         return "%s%s%s" % (self.pfx, self.qty, flv ) 

if __name__ == '__main__':
    pass

    cls = OffTableName

    otn = cls("Dcs", "PmtHv", "Pay" )
    print otn, repr(otn)

    for otn in cls.instances():
        print otn, repr(otn)



