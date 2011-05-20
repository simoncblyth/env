from datetime import datetime, timedelta

from dcs import DcsTableName as DTN
from dcsconf import DCS

from off import OffTableName as OTN   
from offconf import OFF

class Mapping(object):
    """
    Holds table/join level coordinates of source and targets 


    Parameters for the mapping ... which parameter belongs where ?

    #. min interval
    #. max interval
    #. change delta ... does this belong here ?

    Interate over the source to yield dicts to pass to wherever ?	   

    #.  avoid multi-class sources by mapping single class to joined tables ?

    """	
    def __init__(self, dtn, otn):
        self.dtn = dtn
        self.otn = otn   
 

class Scrape(object):
    """
    Database level src and trg
    """ 
    def __init__(self, src, trg ):
        self.src = src
        self.trg = trg

    def __call__(self, mp ):
	"""
	Table level mapping 
	"""
        sfirst = self.src.qa(mp.dtn).first()     
        slast  = self.src.qd(mp.dtn).first()     
        print "sfirst", sfirst
        print "slast", slast

        tfirst  = self.trg.qa(mp.otn).first()
        tlast   = self.trg.qd(mp.otn).first()

        print "tfirst", tfirst
        print "tlast", tlast


if __name__ == '__main__':

    dcs = DCS("dcs")
    off = OFF("recovered_offline_db")
    scr = Scrape( dcs,off )

    ## mappings for scraping into DBI table pair : DcsPmtHv  
    otn = OTN("Dcs","PmtHv","Pay")
    mps = []
    for site in "DBNS".split():
       for det in "AD1 AD2".split():
           for qty in "HV HV_Pw".split():
               dtn = DTN(site, det, qty) 
               mps.append( Mapping(dtn,otn) )

    i = 0 
    while i<1:
        i += 1
        for mp in mps:
            scr(mp)
        #time.sleep(60)


