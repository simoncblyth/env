#!/usr/bin/env python
"""

   ./tgzmon.py --help
   ./tgzmon.py http://dayabay.ihep.ac.cn/data/scm_backup_monitor_SDU.json 
   ./tgzmon.py http://dayabay.phys.ntu.edu.tw/data/scm_backup_monitor_C.json
   ./tgzmon.py http://dayabay.ihep.ac.cn/data/scm_backup_monitor_SDU.json http://dayabay.phys.ntu.edu.tw/data/scm_backup_monitor_C.json

   ./tgzmon.py http://dayabay.ihep.ac.cn/data/scm_backup_monitor_SDU.json  -e blyth@hep1.phys.ntu.edu.tw
   ./tgzmon.py http://dayabay.ihep.ac.cn/data/scm_backup_monitor_SDU.json  -e blyth@hep1.phys.ntu.edu.tw,maqm@ihep.ac.cn

Pull JSON data and apply `monitor_` methods. 
When constraint violations are found send notification emails

Usage from cron 

    cd $ENV_HOME/scm ; python- ; ./tgzmon.py -e blyth@hep1.phys.ntu.edu.tw http://dayabay.ihep.ac.cn/data/scm_backup_monitor_SDU.json 


"""
import os, logging
from datetime import datetime, timedelta
log = logging.getLogger(__name__)
from env.plot.highmon import HighMon

import pytz
utc = pytz.utc
fmt_ = lambda _:_.strftime('%Y-%m-%d %H:%M:%S %Z%z')

def args_():
    from optparse import OptionParser
    parser = OptionParser(usage=__doc__)
    parser.add_option("-l", "--level", default="INFO", help="logging level")
    parser.add_option("-z", "--timezone", default="Asia/Taipei", help="pytz timezone string used for localtime outputs ")
    parser.add_option("-e", "--email",    default="", help="Comma delimited email addresses for notification")
    return parser.parse_args()

class TGZMon(HighMon):
    """
    Reads the JSON plot data, runs `monitor_` methods and 
    sends email in case of violations.

    """
    def __init__(self, *args, **kwa ):
        HighMon.__init__(self, *args, **kwa )
        self.maxage = timedelta(hours=24)   

    def monitor_age(self, method, series ):
        """
        :param method: name 
        :param series: 

        This method (due to its name beginning with 'monitor_' )
        is invoked by the `__call__` on the base class. The 

        The times in the SQLite DB appear to be in CST (naughty naughty boy) hence
        the kludge bringing the timestamp down to UTC to allow that transgression 
        to be sweeped under the carpet. This also allows time handling to follow recommended
        patterns, after the kludge.

        For some good guidelines on python datetime usage, see pytz docs

        * http://pytz.sourceforge.net/

        """
        now_utc = datetime.utcnow().replace(tzinfo=utc)   
        ts = series['data'][-1][0]/1000 - 60*60*8            # 8 hrs kludge 
        dt_utc = datetime.utcfromtimestamp(ts).replace(tzinfo=utc)
        dt_loc = dt_utc.astimezone( loc )
        age = now_utc - dt_utc                         

        log.info("series %-20s ts %s dt_utc %s dt_loc %s age %s " % ( series['name'], ts, fmt_(dt_utc), fmt_(dt_loc), age ))
        if age > self.maxage:
            msg="age %r exceeds maximum allowable %r  " % ( age, self.maxage ) 
            self.add_violation( method=method, series=series['name'], msg=msg )


if __name__ == '__main__':
    opts, args = args_()
    email = opts.email if opts.email else os.environ.get('MAILTO',None)
    loc = pytz.timezone(opts.timezone)   
    logging.basicConfig(level=getattr(logging, opts.level.upper()))
    assert len(args) > 0, "at least one argument URL from which to pull JSON plot data is required " 
    for url in args:
        mon = TGZMon(url, email=email )
        mon()



