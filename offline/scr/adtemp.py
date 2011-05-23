from datetime import datetime, timedelta
from dcs import DcsTableName as DTN, LCR, PTX
from off import OffTableName as OTN   
from scrbase import Mapping, Scrape, SourceSim

class AdTempScrape(Scrape):
    def dtns(cls):
        dtns = []
        for site in ("DBNS",):
            for det in ("SAB",):
                for qty in ("TEMP",):        
                    dtn = DTN(site, det, qty, skipcheck=True)   ## CONVENTION UNKNOWN 
                    dtns.append(dtn)
        return dtns
    dtns = classmethod(dtns) 

    temp_threshold = 0.5      ## GUESS
 
    def __init__(self, srcdb, trgdb, sleep=1):
        """Configure AdTemp scrape, defining mappings between tables/joins in source and target"""
        interval = timedelta(seconds=180)   
        Scrape.__init__(self, sleep)
        target = trgdb.kls(OTN("Dcs","AdTemp","Pay:Vld:SEQNO:V_"))
        for dtn in self.dtns():
            self.append( Mapping(srcdb.kls(dtn),target,interval))
        self.ptx_matcher = PTX()   

    def proceed(self, mapping, update ):
        """
        During a scrape this method is called from the base class, 
        return True if the mapping fulfils significant change or age requirements    
        and the propagate method should be invoked.

        If False is returned then the propagate method is not called on this iteration of the 
        scraper.
        """
        if not mapping.prior:  ## starting up 
            return True
        pd = self._ptxdict( mapping.prior )
        ud = self._ptxdict( update )
        for ptx in sorted(dd.keys()):
            pv = pd[ptx]
            uv = ud[ptx]
            if abs(pv-uv) > temp_threshold:
                return True
        return False

    def propagate(self, inst, cr ):
        dd = self._ptxdict(inst)
        print dd
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


class AdTempSim(SourceSim):
    def __init__(self, srcdb, sleep ):
        SourceSim.__init__(self, sleep )
        for dtn in AdTempScrape.dtns():
            self.append( srcdb.kls(dtn) )
        self.ptx_matcher = PTX()   
    def fake(self, inst, id ):
        """
        Invoked from base class call method, 
        set attributes of source instance to form a fake 

        :param inst: source instance
        :param id: id to assign to the instance instance
        """
        for k,v in inst.asdict.items():
            print k,v 
            if k == 'id':
                setattr( inst, k, id )
            elif k == 'date_time':
                setattr( inst, k, datetime.now() )
            else:
                ptx = self.ptx_matcher(k)
                setattr( inst, k, int(ptx)*10 )    



if __name__ == '__main__':
    from dcsconf import DCS
    from offconf import OFF
    dcs = DCS("dcs")
    off = OFF("recovered_offline_db")
    scr = AdTempScrape( dcs, off , 10 )
    scr(1000)

