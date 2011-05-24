"""
Generic Scraping 

TODO:

#. propa-logging
#. implement scrape testing with 2 processes

#. timezone correction 
#. hook up to NuWa/DybDbi 
#. propagate scrape params from target .spec ? intervals/thresholds/source-tables

#. DAQ ?  ... probably no new DAQ tables coming down pipe, but still needs DBI writing 

"""

import logging

from dcssa import DCS
from offsa import OFF

from pmthv import PmtHvScraper, PmtHvFaker
from adtemp import AdTempScraper, AdTempFaker

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    dcs = DCS("dcs")
    off = OFF("recovered_offline_db")
    scr = PmtHvScraper( dcs, off , 10 )
    scr(max=10)

