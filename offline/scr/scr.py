from datetime import datetime, timedelta

from dcs import DcsTableName as DTN
from dcsconf import DCS

from off import OffTableName as OTN   
from offconf import OFF

class Source(object):
    """
    Interate over the source to yield dicts to pass to the sink ??? 
    Handling multi-class sources ?

    #.  avoid by mapping single class to joined tables ?

    """
    def __init__(self, dcs, dtn ):
        self.dcs = dcs
        self.dtn = dtn
    def gcls(self):
        return self.dcs.lookup( self.dtn )

class Sink(object):
    """
    Drive DybDbi
    """  
    def __init__(self, off, otn ):
        self.off = off
        self.otn = otn

class Mapping(object):
    """
    Parameters for the mapping OR source OR sink ... which parameter belongs where ?

    #. min interval
    #. max interval
    #. change delta ... does this belong here ?

    """
    def __init__(self, source , sink ):
        pass      


class Scraper(dict):
    """
    pseudo-code translation of the old scraper, in order to see what needs
    to be factored out 
    """
    interval = timedelta(seconds=20)   ## needs to be property of a mapping class ?
    def __init__(self, dcs, off):

        lvld = off.qd(Vld).first()   
        if lvld == None:
            next = 0
        else:
            next = lvld.TIMESTART + self.interval
        self['next'] = next 
        self['recursion'] = 0

        self.dcs = dcs
        self.off = off

    def __call__(self, clsa, clsb ):
        """ 
        TODO:

        #. encapulate the 2 queries into one on a join ... so 
           can tease out the table specifics into one mapped class ... improving 
           genericness

        """ 
        self['recursion'] += 1
        ia = self.dcs.qa(clsa).filter(clsa.date_time >= self['next']).first() 
        if ia == None:
            return
        assert ia.id > 0                

        ib = self.dcs.qa(clsb).filter(clsb.id == ia.id).first()
        if ib == None:
            print "no related clsb %s record for %r " % ( clsb.__name__, ia )
        
        ## this stuff needs to be DybDbied anyhow        
        timeStart = ia.date_time
        timeEnd = timeStart + self.interval
        siteMask = 
        subsite = 

        self['next'] = timeEnd




if __name__ == '__main__':

    ## need way to address joins

    dcs = DCS("dcs")
    dtn = DTN("DBNS","AD1","HV")  
    src = Source( dcs , dtn )

    off = OFF("recovered_offline_db")
    otn = OTN("Dcs","PmtHv","Pay")
    snk = Sink( off, otn )

    scr = Scraper( src, snk )

    for site in "DBNS".split():
        for det in "AD1 AD2".split():
            dhv = dcs.lookup( DTN( site, det, "HV" )    )
            dpw = dcs.lookup( DTN( site, det, "HV_Pw" ) )

    scr()



