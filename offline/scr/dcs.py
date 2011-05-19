from sa import sa_meta, sa_url
from sqlalchemy import MetaData, create_engine
from sqlalchemy.sql import join
from sqlalchemy.orm import sessionmaker, mapper
Session = sessionmaker()


class SA(object):
    def __init__(self, dbconf):
        meta = MetaData()
        engine = create_engine( sa_url(dbconf), echo=False )
        meta.reflect(bind=engine)    

        Session.configure(bind=engine)
        session = Session()

        self.meta = meta
        self.session = session

    def add(self, obj):
        self.session.add( obj )

    def commit(self):
        self.session.commit()  


class DcsTable(object):
    """
    DCS table name provider, usage ::

         dcstn = Dcs("DBNS", "AD1" )
         print dcstn.hv

    """
    siteList = ["DBNS", "LANS", "FARS", "MIDS" "Aberdeen", "SAB"]
    detList = ["Unknown", "AD1", "AD2", "AD3", "AD4", "IWS", "OWS", "RPC" ]
    def __init__(self, site, det ):
        assert site in self.siteList, "site \"%s\" is not in \"%r\"" % ( site , self.siteList ) 
        assert det  in self.detList, "detector \"%s\" is not in \"%s\"" %  ( det , self.detList ) 
        self.site = site
        self.det = det 

    def lcr(self, qty="HV" ):
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

    def __getattr__(self, att):
         
     


    ## sqlalchemy.schema.Table for LCR tables
    hv    = property(lambda self:self.meta.tables[self.lcr("HV")]) 
    pw    = property(lambda self:self.meta.tables[self.lcr("HV_Pw")]) 
    ## no FK so must specify the onclause
    hp    = property(lambda self:join( self.hv, self.pw , self.hv.c.id == self.pw.c.id ))  


class Hv(object):
    pass

class Pw(object):
    pass

def lcr():
    for l in range(8,0,-1):
        for c in range(3,0,-1):
            for r in range(8,0,-1):
                yield l,c,r


if __name__ == '__main__':



    dcs = SA("dcs")


    #ds = LCR("DBNS", "AD1" )
    ds = LCR("DBNS", "AD2" )
    #ds = LCR("SAB", "AD1" )
    #ds = LCR("SAB", "AD2" )
    print ds.hv.__class__
    print ds.pw.__class__
    print ds.hp.__class__

    mapper( Hv ,  ds.hv ) 
       
    fake = lambda _:_[0]*100 + _[1]*10 + _[2] 
    attr = lambda _:"L%dC%dR%d" % ( _[0],_[1],_[2] )

    import re
    from datetime import datetime

    kptn = re.compile("^L(?P<l>\d)C(?P<c>\d)R(?P<r>\d)$")

    hv = Hv()    
   
    for k in ds.hv.c.keys():
        m = kptn.match(k)
        if m:
            d = m.groupdict()
            lcr = map(int, (d['l'],d['c'],d['r']))
            fk  = fake(lcr)
            print d,lcr,fk
            setattr( hv , k , fk )
        elif k == 'id':
            setattr( hv , k , 1 )
        elif k == 'date_time':
            setattr( hv , k , datetime.now() )
            

    ds.add(hv)
    ds.commit()


