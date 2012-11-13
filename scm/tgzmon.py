#!/usr/bin/env python
"""


"""

import logging
from datetime import datetime, timedelta
log = logging.getLogger(__name__)
from env.plot.highmon import HighMon

import pytz
utc = pytz.utc
loc = pytz.timezone('Asia/Taipei')   # make configurable ?
fmt_ = lambda _:_.strftime('%Y-%m-%d %H:%M:%S %Z%z')

class TGZMon(HighMon):
    """
    """
    def __init__(self, *args, **kwa ):
        HighMon.__init__(self, *args, **kwa )
        self.maxage = timedelta(hours=2)   

    def monitor_age(self, method, series ):
        """
        :param method: name 
        :param series: 

        The times in the SQLite DB appear to be in CST (naughty naughty boy) hence
        the kludge bringing the timestamp down to UTC to allow that transgression 
        to be sweeped under the carpet. This also allows time handling to follow recommended
        patterns, after the kludge.

        """
        now_utc = datetime.utcnow().replace(tzinfo=utc)   
        ts = series['data'][-1][0]/1000 - 60*60*8            # mysterious 8 hrs kludge 
        dt_utc = datetime.utcfromtimestamp(ts).replace(tzinfo=utc)
        dt_loc = dt_utc.astimezone( loc )
        age = now_utc - dt_utc                         

        log.info("series %-20s ts %s dt_utc %s dt_loc %s age %s " % ( series['name'], ts, fmt_(dt_utc), fmt_(dt_loc), age ))
        if age > self.maxage:
            msg="age %r exceeds maximum allowable %r  " % ( age, self.maxage ) 
            self.add_violation( method=method, series=series['name'], msg=msg )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    url = "http://dayabay.ihep.ac.cn/data/scm_backup_monitor_SDU.json"
    #url = "http://dayabay.phys.ntu.edu.tw/data/scm_backup_monitor_C.json"
    mon = TGZMon(url, email="blyth@hep1.phys.ntu.edu.tw" )
    mon()
    assert len(mon) == 0 , repr(mon) 



