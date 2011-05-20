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

from dcs import DcsTableName as DTN, LCR
from dcsconf import DCS
from off import OffTableName as OTN   
from offconf import OFF

class Mapping(object):
    """
    ``Mapping`` instances represent the association between 
    a single table/join in the source DB to a single table/join 
    in the target DB

    Parameters of mapping make most sense to 
    to carried on the target DybDbi class ? and maintained
    as meta entries in the .spec

    #. min interval
    #. max interval
    #. change delta threshold 

    """	
    interval = timedelta(seconds=60)   ## interval on mapping OR scrape ?
    def __init__(self, source, target):
        """
        Specify source and target SA classes 
        (no need to use DybDbi target class and Rpt for just reading a Vld 
        BUT ... do need the DybDbi target class and writer for valid DBI
        writing )
        """
        self.source = source
        self.target = target
        self.next = self.target_next()

    def target_next(self):
        """
        Return time of last entry in target plus the update interval, 
        which corresponds to earliest source eligibility cutoff
        """
        tlast = self.target.last()
        return 0 if tlast == None else tlast.TIMESTART + self.interval

    def __call__(self):
        """
        Check for eligible entries in the source, when found
        propagate the last(?) to target 
        """
        eligible = self.source.qafter(self.next).first()   
        if eligible == None:
            print "no src entry after %s " % self.next
            return 
        self.propagate(eligible)

    def propagate(self, srci ):
        """
        Propagate src instance to target ...
   
        TODO:
        #. grab relevant DybDbi .spec class, configure writer 

        """
        print "propagate src instance %r " % srci
        cr = srci.contextrange(self.interval)
        print cr
        self.next = cr['timeEnd']

        ## hmm where to put this junk ... it is too specific to belong here 
        for (l,c,r),v in sorted(srci.lcrdict.items()):
            #print l,c,r,v['hv'],v['pw']
            pass


if __name__ == '__main__':

    ## this config needs capturing in a class ?
    
    off = OFF("recovered_offline_db")
    otn = OTN("Dcs","PmtHv","Pay:Vld:SEQNO:V_")   ## auto-join
    target = off.kls( otn )                      

    dcs = DCS("dcs")
    mps = []
    for site in "DBNS".split():
       for det in "AD1".split():
           for qty in "HV:HV_Pw:id:P_".split():
               dtn = DTN(site, det, qty)
               source = dcs.kls(dtn)
               mps.append( Mapping(source,target) )

    i = 0 
    while i<3:
        i += 1
        for mp in mps:
            mp()
        time.sleep(1)





