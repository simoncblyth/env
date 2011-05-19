from datetime import datetime, timedelta
from dcs import Hv, Pw
from off import Vld      ## hmmm need to sort out dynamic class creation/naming

from dcsconf import DCS
from offconf import OFF

class Scraper(dict):
    interval = timedelta(seconds=20)   ## needs to be property of mapped class ?
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

    def __call__(self):
        """ 
        TODO:

        #. encapulate the 2 queries on Hv and Pw into one on a join ... so 
           can tease out the table specifics into one mapped class ... improving 
           genericness

        """ 

        self['recursion'] += 1 
        ihv = self.dcs.qa(Hv).filter(Hv.date_time >= self['next']).first() 
        if ihv == None:
            return
        assert ihv.id > 0                

        ipw = self.dcs.qa(Pw).filter(Pw.id == ihv.id).first()
        if ipw == None:
            print "no related Pw record for Hv %r " % ihv
                
        timeStart = ihv.date_time
        timeEnd = timeStart + self.interval
        siteMask = 
        subsite = 

        self['next'] = timeEnd



if __name__ == '__main__':

    dcs = DCS("dcs")
    off = OFF("recovered_offline_db")

    for gcln, gcls in dcs.classes.items():


    scr = Scraper( dcs, off )
    scr()

    #last = dcs.qd(Hv).first()



