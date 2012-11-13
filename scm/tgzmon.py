#!/usr/bin/env python
"""

TODO:

TZ debugging

simon:scm blyth$ ./tgzmon.py 
DEBUG:__main__:utcnow 2012-11-13 05:49:06.341120 
DEBUG:__main__:stamp  2012-11-13 08:00:00 
DEBUG:__main__:method monitor_age name OK stamp 2012-11-13 08:00:00 age -1 day, 21:49:06.341120 
DEBUG:__main__:utcnow 2012-11-13 05:49:06.341120 
DEBUG:__main__:stamp  2012-11-13 20:00:01 
DEBUG:__main__:method monitor_age name svn/dybaux stamp 2012-11-13 20:00:01 age -1 day, 9:49:05.341120 
DEBUG:__main__:utcnow 2012-11-13 05:49:06.341120 
DEBUG:__main__:stamp  2012-11-13 20:00:01 
DEBUG:__main__:method monitor_age name svn/dybsvn stamp 2012-11-13 20:00:01 age -1 day, 9:49:05.341120 
DEBUG:__main__:utcnow 2012-11-13 05:49:06.341120 
DEBUG:__main__:stamp  2012-11-13 20:00:01 
DEBUG:__main__:method monitor_age name tracs/dybaux stamp 2012-11-13 20:00:01 age -1 day, 9:49:05.341120 
DEBUG:__main__:utcnow 2012-11-13 05:49:06.341120 
DEBUG:__main__:stamp  2012-11-13 20:00:01 
DEBUG:__main__:method monitor_age name tracs/dybsvn stamp 2012-11-13 20:00:01 age -1 day, 9:49:05.341120 
INFO:env.plot.highmon:no violations, not sending email

"""

import logging
from datetime import datetime, timedelta
log = logging.getLogger(__name__)
from env.plot.highmon import HighMon

class TGZMon(HighMon):
    """
    """
    def __init__(self, *args, **kwa ):
        HighMon.__init__(self, *args, **kwa )
        self.utcnow = datetime.utcnow()
        self.maxage = timedelta(hours=24)   

    def monitor_age(self, method, series ):
        """
        :param method: name 
        :param series: 

        Timestamps are UTC ms since epoch, the age is obtained from the 
        delta of two naive UTC datetimes
        """
        stamp = datetime.fromtimestamp( series['data'][-1][0]/1e3 )  
        age = self.utcnow - stamp                         
        log.debug("utcnow %s " % self.utcnow )
        log.debug("stamp  %s " % stamp  )
        log.debug("method %s name %s stamp %s age %s " % ( method, series['name'], stamp, age ))
        if age > self.maxage:
            msg="age %r exceeds maximum allowable %r  " % ( age, self.maxage ) 
            self.add_violation( method=method, name=series['name'], msg=msg )


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    url = "http://dayabay.ihep.ac.cn/data/scm_backup_monitor_SDU.json"
    mon = TGZMon(url, email="blyth@hep1.phys.ntu.edu.tw" )
    mon()
    assert len(mon) == 0 , repr(mon) 



