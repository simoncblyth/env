from datetime import datetime, timedelta

from scrbase import Mapping, Scraper, Faker
from dcs import DcsTableName as DTN, LCR, PTX
from off import OffTableName as OTN   

class PmtHvScraper(Scraper):
    def dtns(cls):    
        """List of source tables/joins of interest """
        dtns = []
        for site in ("DBNS",):
            for det in ("AD1",):
                for qty in ("HV:HV_Pw:id:P_",):            ## auto-join
                    dtns.append(DTN(site, det, qty))
        return dtns 
    dtns = classmethod(dtns) 
    voltage_threshold = 1.0

    def __init__(self, srcdb, trgdb, sleep ):
        """
        Configure PmtHv scrape, defining mappings between tables/joins in source and target 

        :param srcdb: dbconf sectname of source database
        :param trgdb: dbconf sectname of target database
        :param sleep: seconds to sleep between scrape interations
        """
        Scraper.__init__(self, sleep)
        target = trgdb.kls(OTN("Dcs","PmtHv","Pay:Vld:SEQNO:V_"))   ## auto-join
        interval = timedelta(seconds=60)   
        for dtn in self.dtns():
            self.append( Mapping(srcdb.kls(dtn),target,interval))
        self.lcr_matcher = LCR()   

    def changed(self, prev, curr ):
        """ Decide if reason to proceed to propagate """
        p = self._lcrdict( prev )
        c = self._lcrdict( curr )
        for lcr in sorted(p.keys()):
            pv = p[lcr]
            cv = c[lcr]
            if abs(pv['pw']-cv['pw']) > 0:
                return True
            if abs(pv['voltage']-cv['voltage']) > voltage_threshold:
                return True
        return False

    def propagate(self, inst , tcr ):
        """
        During a scrape this method is called from the base class,
        return True in order to move on with a mapping (incrementing the nexttime )

        :param inst: source instance 
        :param tcr: target context range
 
        TODO: add DybDbi writing

        from DybDbi import GDcsPmtHv
        wrt = GDcsPmtHv.Wrt()
        wrt.ctx( contextrange=tcr , ...)
        wrt.Write( **d )
        wrt.Close() 
        """
        dd = self._lcrdict(inst)
        n = 0
        for (l,c,r),v in sorted(dd.items()):
            d = dict(ladder=l,col=c,ring=r,voltage=v['voltage'],pw=v['pw'])
            n += 1
        print "%s propagate source inst %r tcr %r into %d target dicts " % (self.__class__.__name__, inst, tcr, n )
        return True  

    def _lcrdict(self, inst):
        """Examines source instance, extracting ladder,col,ring and values and collecting into a dict.""" 
        dd = {}
        for k,v in inst.asdict.items():
            if k in 'P_id id date_time P_date_time'.split():
                continue 
            qty,kk = ('pw',k[2:]) if k.startswith('P_') else ('voltage',k)
            lcr = self.lcr_matcher(kk)  
            if dd.has_key(lcr):
                dd[lcr].update({qty:v})
            else:
                dd[lcr] = {qty:v}
        return dd


class PmtHvFaker(Faker):
    """
    Creates fake instances and inserts them into sourcedb   
    """
    def __init__(self, srcdb, sleep ):
        Faker.__init__(self, sleep )
        for dtn in PmtHvScraper.dtns():
            self.append( srcdb.kls(dtn) )
        self.lcr_matcher = LCR()   

    def fake(self, inst, id ):
        """
        Invoked from base class call method, 
        set attributes of source instance to form a fake 

        :param inst: source instance
        :param id: id to assign to the instance instance
        """
        fakefn=lambda (l,c,r),qty:l*100 + c*10 + r if qty == "voltage" else 1 
        for k,v in inst.asdict.items():
            if k in 'id P_id'.split():
                 setattr( inst, k, id )
            elif k in 'date_time P_date_time'.split():
                 setattr( inst, k, datetime.now() )
            else:
                qty,kk = ('pw',k[2:]) if k.startswith('P_') else ('voltage',k)
                lcr = self.lcr_matcher(kk)    
                setattr( inst , k, fakefn( lcr, qty) )


if __name__ == '__main__':
    pass 
    from dcssa import DCS
    from offsa import OFF
    dcs = DCS("dcs")
    off = OFF("recovered_offline_db")
    scr = PmtHvScraper( dcs, off , 10 )
    scr(1000)


