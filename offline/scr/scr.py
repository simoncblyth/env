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

from scrbase import Mapping, Scrape, SourceSim

class PmtHvScrape(Scrape):
    def dtns(cls):    
        """
        List of source tables/joins of interest 
        """
        dtns = []
        for site in ("DBNS",):
            for det in ("AD1",):
                for qty in ("HV:HV_Pw:id:P_",):            ## auto-join
                    dtns.append(DTN(site, det, qty))
        return dtns 
    dtns = classmethod(dtns) 

    def __init__(self, srcdb, trgdb, sleep ):
        """
        Configure PmtHv scrape, defining mappings between tables/joins in source and target 

        :param srcdb: dbconf sectname of source database
        :param trgdb: dbconf sectname of target database
        :param sleep: seconds to sleep between scrape interations

        """
        Scrape.__init__(self, sleep)
        target = trgdb.kls(OTN("Dcs","PmtHv","Pay:Vld:SEQNO:V_"))   ## auto-join
        interval = timedelta(seconds=60)   
        for dtn in self.dtns():
            self.append( Mapping(srcdb.kls(dtn),target,interval))
        self.lcr_matcher = LCR()   

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
        print "propagate inst %r tcr %r " % (inst, tcr)
        dd = self._lcrdict(inst)
        for (l,c,r),v in sorted(dd.items()):
            d = dict(ladder=l,col=c,ring=r,voltage=v['voltage'],pw=v['pw'])
            print d
        return True  

    def _lcrdict(self, inst):
        """
        Examine source instance, extracting ladder,col,ring and values
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


class PmtHvSim(SourceSim):
    """
    Creates fake instances and feeds them to sourcedb   
    """
    def __init__(self, srcdb, sleep ):
        SourceSim.__init__(self, sleep )
        for dtn in PmtHvScrape.dtns():
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

