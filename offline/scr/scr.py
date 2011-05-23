"""
Generic Scraping 

TODO:

#. extend to Temp ... specifics encapsulation 
#. get the dcssim operational
#. implement scrape testing with 2 processes
#. hook up to NuWa/DybDbi 
#. DAQ ?  ... probably no new DAQ tables coming down pipe, but still needs DBI writing 

"""
import time
from datetime import datetime, timedelta

from dcs import DcsTableName as DTN, LCR, PTX
from dcsconf import DCS
from off import OffTableName as OTN   
from offconf import OFF

from scrbase import Mapping, Scrape

class PmtHvScrape(Scrape):
    def __init__(self, srcdb, trgdb, sleep=1):
        """
        Configure PmtHv scrape, defining mappings between tables/joins in source and target 

        :param srcdb: dbconf sectname of source database
        :param trgdb: dbconf sectname of target database
        :param sleep: seconds to sleep between scrape interations

        """
        Scrape.__init__(self, sleep)
        target = trgdb.kls(OTN("Dcs","PmtHv","Pay:Vld:SEQNO:V_"))   ## auto-join

        interval = timedelta(seconds=60)   
        for site in ("DBNS",):
            for det in ("AD1",):
                for qty in ("HV:HV_Pw:id:P_",):            ## auto-join
                    source = srcdb.kls(DTN(site, det, qty))
                    self.append( Mapping(source,target,interval))
        self.lcr_matcher = LCR()   

    def propagate(self, inst , tcr ):
        """
        :param inst: source instance 
        :param tcr: target context range

        TODO: add DybDbi writing

        from DybDbi import GDcsPmtHv
        wrt = GDcsPmtHv.Wrt()
        wrt.ctx( contextrange=tcr , ...)
        wrt.Write( **d )
        wrt.Close() 
        """

        print "propagate_ src inst %r tcr %r " % (inst, tcr)
        dd = self._lcrdict(inst)
        for (l,c,r),v in sorted(dd.items()):
            d = dict(ladder=l,col=c,ring=r,voltage=v['voltage'],pw=v['pw'])
            print d
        return True   ## NB MUST RETURN True TO MOVE ON WITH THE MAPPING
    def _lcrdict(self, inst):
        """
        Juice the source instance, extracting ladder,col,ring and values
        and collecting into a dict. 
        """ 
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


class AdTempScrape(Scrape):
    def __init__(self, srcdb, trgdb, sleep=1):
        """
        Configure AdTemp scrape, defining mappings between tables/joins in source and target 
        """
        interval = timedelta(seconds=180)   
        Scrape.__init__(self, sleep)
        target = trgdb.kls(OTN("Dcs","AdTemp","Pay:Vld:SEQNO:V_"))   ## auto-join
        for site in ("DBNS",):
            for det in ("SAB",):
                for qty in ("TEMP",):        
                    self.append( Mapping( srcdb.kls(DTN(site, det, qty)),target,interval))
        self.ptx_matcher = PTX()   
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
                dd['Temp_%s'%ptx] = v
        return dd 


 
if __name__ == '__main__':
    import sys
    dcs = DCS("dcs")
    off = OFF("recovered_offline_db")

    scr = PmtHvScrape( dcs, off )
    scr()

