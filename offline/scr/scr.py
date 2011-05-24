"""
Generic Scraping 

TODO:

#. delta-ing ...age checking 
#. propa-logging
#. implement scrape testing with 2 processes
#. hook up to NuWa/DybDbi 
#. DAQ ?  ... probably no new DAQ tables coming down pipe, but still needs DBI writing 

"""

from dcssa import DCS
from offsa import OFF

from pmthv import PmtHvScrape, PmtHvSim
from adtemp import AdTempScrape, AdTempSim
 
if __name__ == '__main__':
    import sys
    dcs = DCS("dcs")
    off = OFF("recovered_offline_db")
    scr = PmtHvScrape( dcs, off , 10 )
    scr()

