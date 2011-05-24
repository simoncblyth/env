"""
Scrape features to test

#. no-significant change, yields no update (except 1st run OR age_threshold)

How to test a scrape ? Granularity choice ?
#. full set of iterations  (simpler than iteration-by-iteration, which would entail interprocess communication) 

#. main process launches updater/scraper sub-processes (configured with clear time-to-live/max iterations)
   #. update simulation (with knows sequence of entries, designed to tickle features)
   #. scraper
#. main process blocks on joining those ... then examines tealeaves for expectations 

#. test specific parameters (for intervals/sleeps) in order for test to complete in reasonable time



#. RERUNABILITY .. START FROM STANDARD STATE ?


"""

import logging
log = logging.getLogger(__name__)


from dcssa import DCS
from offsa import OFF
from scr import PmtHvScraper, PmtHvFaker, AdTempFaker, AdTempScraper

from multiprocessing import Process

def runfkr(cfg):
    log.debug(cfg)
    dcs = DCS("dcs")
    fkr = AdTempFaker(dcs, sleep=cfg['sleep'] )
    fkr(max=cfg['max'])   

def runscr(cfg):
    log.debug(cfg)
    dcs = DCS("dcs")
    off = OFF("recovered_offline_db")
    scr = AdTempScraper( dcs, off , sleep=cfg['sleep'] )
    scr(max=cfg['max'])


if __name__ == '__main__':
    pass

    logging.basicConfig(level=logging.INFO)

    cfg = dict(sleep=5, max=3)
    fkr = Process(target=runfkr, args=(cfg,))
    scr = Process(target=runscr, args=(cfg,))

    fkr.start()
    scr.start()

    ## join blocks until sub-process completes
    fkr.join()   
    scr.join()





