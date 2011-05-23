import re

class LCR(object):
    kptn = re.compile("^L(?P<ladder>\d)C(?P<col>\d)R(?P<ring>\d)$")
    def __call__(self, k):
        m = self.kptn.match(k)
        if m:
            d = m.groupdict()
            return (int(d['ladder']),int(d['col']),int(d['ring']),)
        return None

class PTX(object):
    kptn = re.compile("^.*_PT(?P<ptx>\d)$")
    def __call__(self, k):
        m = self.kptn.match(k)
        if m:
            d = m.groupdict()
            return d['ptx']
        return None


class DcsTableName(object):
    """
    DCS table name provider, returning names of tables based on 
    ctor (site, det, qty) arguments usage::

         from dcs import DcsTableName as DTN
         dtn = DTN("DBNS", "AD1", "HV" )
         print dtn
         print dtn.site, dtn.det, dtn.qty
         print dtn.gcln    ## name of dynamic class for a single table mapping 
      

    For example the names of LCR (ladder/column/ring) tables
    can be returned, for details see oum:sop/dcs
 
           =================  ==============================   ===========================
             LCR table          description                      notes
           =================  ==============================   ===========================
            DBNS_AD1_HV        oum:sop/dcs/#dbns-ad1-hv
            DBNS_AD2_HV        oum:sop/dcs/#dbns-ad2-hv
            SAB_AD1_HV_Vmon    oum:sop/dcs/#sab-ad1-hv-vmon
            SAB_AD2_HV_Vmon    oum:sop/dcs/#sab-ad2-hv-vmon
            DBNS_AD1_HVPw      oum:sop/dcs/#dbns-ad1-hvpw       non-uniform naming
            SAB_AD2_HV_Pw      oum:sop/dcs/#sab-ad2-hv-pw
            DBNS_AD2_HV_Pw     oum:sop/dcs/#dbns-ad2-hv-pw
            SAB_AD1_HV_Pw      oum:sop/dcs/#sab-ad1-hv-pw
           =================  ==============================   ===========================

    valid qty are HV, HV_Pw, HV_Vmon 

    Extending this to temperatures is clear as mud, as only oum:sop/dcs/#dbns-sab-temp has
    the _PT1/2/3/4/5 the code points to using by that table name does not follow the convention

    """
    siteList = ["DBNS", "LANS", "FARS", "MIDS", "Aberdeen", "SAB"]
    detList = ["Unknown", "AD1", "AD2", "AD3", "AD4", "IWS", "OWS", "RPC" ]
    qtyList = ["HV","HV_Pw", "HV_Vmon" ]

    kln = property(lambda self:'G%s_%s_%s' % ( self.site, self.det, self.qty ))     #  rationalized class name G<site>_<det> 


    ## convert from DCS to offline conventions ... TODO: use the enum when PORT TO NuWa
    def _sitemask(self, site):
        return 1 << self.siteList.index(site)
    def _subsite(self, det):
        if self.qty == "TEMP": 
            return 1
        return self.detList.index(det)
    sitemask = property(lambda self:self._sitemask(self.site))
    subsite  = property(lambda self:self._subsite(self.det))


    def __init__(self, site, det, qty , skipcheck=False):
        self.site = site
        self.det = det 
        self.qty = qty
        if skipcheck:   ## for table names that do not play by the conventions 
            return

        assert site in self.siteList, "site \"%s\" is not in \"%r\"" % ( site , self.siteList ) 
        assert det  in self.detList, "detector \"%s\" is not in \"%r\"" %  ( det , self.detList ) 
        assert qty in self.qtyList or self.isjoin, "qty \"%s\" is not in \"%r\"" % ( qty, self.qtyList )

    isjoin = property(lambda self:self.qty.find(":") > -1)
    def jbits(self):
       assert self.isjoin
       a,b,j,tb = self.qty.split(":")   ## tb is tiebreaker prefix for name collisions
       xa = DcsTableName( self.site, self.det, a )
       xb = DcsTableName( self.site, self.det, b )
       return xa,xb,j,tb

    def __repr__(self):
        return "DTN %-15s %-10s %-10s %-10s     %-10s" % ( str(self), self.site, self.det, self.qty, self.kln )

    def __str__(self):
        """
        correct what looks like a bug in table naming, on rendering 
        """
        if self.site == "DBNS" and self.det == "AD1" and self.qty == "HV_Pw":
            qty = "HVPw"      
        else:
            qty = self.qty
        return "%s_%s_%s" % ( self.site, self.det, qty ) 

 
def lcr():
    for l in range(8,0,-1):
        for c in range(3,0,-1):
            for r in range(8,0,-1):
                yield l,c,r


def instances():
    """Selection of DcsTableName instances"""
    dtns = []
    for site in "DBNS".split():
        for det in "AD1 AD2".split():
            for qty in "HV HV_Pw".split():
                dtns.append( cls(site, det, qty) )
    return dtns
def tables():    
    return map(str, instances() )
def classes():
    return map(lambda _:_.kln, instances() )

            
if __name__ == '__main__':
            
    cls = DcsTableName

    print "instances",  instances()
    print "tables",    tables()
    print "classes", classes()

    dtn = cls("DBNS", "AD1", "HV" )
    print dtn, repr(dtn)

    for dtn in instances():
        print dtn, repr(dtn)

