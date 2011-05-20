from datetime import datetime, timedelta

from dcs import DcsTableName as DTN
from dcsconf import DCS

from off import OffTableName as OTN   
from offconf import OFF

class Mapping(object):
    """
    A ``Mapping`` instance represents the association between 
    a single table/join in the source db to a single table in
    the target db

    Parameters for the mapping ... which parameter belongs where ?

    #. min interval
    #. max interval
    #. change delta ... does this belong here ?

    Interate over the source to yield dicts to pass to wherever ?	   

    #.  avoid multi-class sources by mapping single class to joined tables ?

    """	
    interval = timedelta(seconds=60)   ## interval on mapping OR scrape ?
    def __init__(self, source, target):
        self.source = source
        self.target = target

    def target_next(self):
        tlast = self.target.last()
        print tlast
        if tlast == None:
            return 0
        else:
            return tlast.TIMESTART + self.interval 

class Scrape(object):
    """
    Hmm ... `Scrape` as a list of `Mapping` ??

    ## below lines parameterize the scrape at table level ... captured into a class ?
    ## mappings for scraping into DBI table pair : DcsPmtHv  

    """ 
    def __init__(self):
        pass

    def __call__(self, mp ):
	"""
	Table level mapping 
	"""
        print "sfirst", mp.source.first()
        print "slast", mp.source.last()

        print "tfirst", mp.target.first()
        print "tlast", mp.target.last()

        next = mp.target_next()
        print "next %s " % next



if __name__ == '__main__':
    
    off = OFF("recovered_offline_db")
    otn = OTN("Dcs","PmtHv","Pay:Vld:SEQNO")
    tkls = off.kls( otn )   ## target kls

    dcs = DCS("dcs")
    mps = []
    for site in "DBNS".split():
       for det in "AD1 AD2".split():
           for qty in "HV HV_Pw".split():
               dtn = DTN(site, det, qty)
               scls = dcs.kls(dtn)
               mps.append( Mapping(scls,tkls) )

    scr = Scrape()

    i = 0 
    while i<1:
        i += 1
        for mp in mps:
            scr(mp)
        #time.sleep(60)


