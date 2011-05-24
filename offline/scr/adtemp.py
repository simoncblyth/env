
from datetime import datetime, timedelta
from dcs import DcsTableName as DTN, LCR, PTX
from off import OffTableName as OTN   
from scrbase import Mapping, Scraper, Faker

import logging
log = logging.getLogger(__name__)


class AdTempScraper(Scraper):
    def dtns(cls):
        """Coordinates of source table/joins"""
        dtns = []
        for site in ("DBNS",):
            for det in ("SAB",):
                for qty in ("TEMP",):        
                    dtn = DTN(site, det, qty, skipcheck=True)   ## CONVENTION UNKNOWN 
                    dtns.append(dtn)
        return dtns
    dtns = classmethod(dtns) 
 
    temp_threshold = 0.5                ## GUESS
    interval = timedelta(minutes=3)     ## validity interval
  
    def __init__(self, srcdb, trgdb, sleep=1):
        """Configure AdTemp scrape, defining mappings between tables/joins in source and target"""
        Scraper.__init__(self, sleep)
        target = trgdb.kls(OTN("Dcs","AdTemp","Pay:Vld:SEQNO:V_"))
        for dtn in self.dtns():
            self.append( Mapping(srcdb.kls(dtn),target,self.interval))
        self.ptx_matcher = PTX()   

    def changed(self, prev, curr ):
        """Overrides base class, returns decision to proceed to propagate"""
        pd = self._ptxdict( prev )
        ud = self._ptxdict( curr )
        log.debug("prev %s " % pd )
        log.debug("curr %s " % ud )
        for ptx in sorted(pd.keys()):
            pv = float(pd[ptx])    ## often decimal.Decimal types
            uv = float(ud[ptx])
            df = abs(pv-uv) 
            log.debug("pv %r uv %r df %r " % (pv,uv,df)) 
            if df > self.temp_threshold:
                return True
        return False

    def propagate(self, curr, contextrange ):
        """Overrides base class, performs propagation"""
        dd = self._ptxdict(curr)
        log.info("propagate %r " % dd)
        return True

    def _ptxdict(self, inst ):
        dd = {}
        for k,v in inst.asdict.items():
            if k in 'id date_time'.split():
                continue
            ptx = self.ptx_matcher(k)
            if ptx:   
                dd['Temp_PT%s'%ptx] = v
        return dd 



class AdTempFaker(Faker):
    def __init__(self, srcdb, sleep ):
        Faker.__init__(self, sleep )
        for dtn in AdTempScraper.dtns():
            self.append( srcdb.kls(dtn) )
        self.ptx_matcher = PTX()   
    def fake(self, inst, id ):
        """
        Invoked from base class, sets source instance to form a fake 
        :param inst: source instance
        :param id: id to assign to the instance 
        """
        for k,v in inst.asdict.items():
            if k == 'id':
                setattr( inst, k, id )
            elif k == 'date_time':
                setattr( inst, k, datetime.now() )
            else:
                ptx = self.ptx_matcher(k)
                setattr( inst, k, int(ptx)*10 )    



if __name__ == '__main__':
    from dcssa import DCS
    from offsa import OFF
    dcs = DCS("dcs")
    off = OFF("recovered_offline_db")
    scr = AdTempScraper( dcs, off , 10 )
    scr(1000)

