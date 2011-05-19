import re
from datetime import datetime


class DcsTableName(object):
    """
    DCS table name provider, returing the names of tables based on 
    ctor (site, det) arguments and qty attribute name, usage::

         tn = DcsTableName("DBNS", "AD1" )
         print tn.hv
         print tn.pw

    """
    siteList = ["DBNS", "LANS", "FARS", "MIDS", "Aberdeen", "SAB"]
    detList = ["Unknown", "AD1", "AD2", "AD3", "AD4", "IWS", "OWS", "RPC" ]
    def __init__(self, site, det ):
        assert site in self.siteList, "site \"%s\" is not in \"%r\"" % ( site , self.siteList ) 
        assert det  in self.detList, "detector \"%s\" is not in \"%s\"" %  ( det , self.detList ) 
        self.site = site
        self.det = det 

    lcr_qtys = dict(hv="HV",pw="HV_Pw")
    def lcr(self, qty ):
        """
        Returns the names of LCR (ladder/column/ring) tables, for  
        examples of LCR tables see oum:sop/dcs
 
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

        """
        if self.site == "DBNS" and self.det == "AD1" and qty == "HV_Pw":
            qty = "HVPw"      ## correct what looks like a bug in table naming  
        return "%s_%s_%s" % ( self.site, self.det , qty )     

    def __getattr__(self, name):
        if name in self.lcr_qtys:
            return self.lcr( self.lcr_qtys[name] )
        else:
            raise AttributeError

def lcr():
    for l in range(8,0,-1):
        for c in range(3,0,-1):
            for r in range(8,0,-1):
                yield l,c,r





class BaseT(object):
    """
    Base for mapped classes that have a date_time attribute
    """
    def __repr__(self):
        return "%s %s %s " % ( self.__class__.__name__, self.id, self.date_time )

class Hv(BaseT):
    pass
class Pw(BaseT):
    pass


            
if __name__ == '__main__':
            
    tn = DcsTableName("DBNS", "AD1" )
    print tn.hv
    print tn.pw

    for site in DcsTableName.siteList:
        for det in DcsTableName.detList:
            tn = DcsTableName(site, det)   ## many will be non-sensical
            print tn.hv
            print tn.pw

