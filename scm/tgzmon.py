#!/usr/bin/env python
"""

   ./tgzmon.py --help
   ./tgzmon.py        ##  defaults to checking a standard set of JSON URLs which depend on the NODE_TAG 
   ./tgzmon.py http://dayabay.ihep.ac.cn/data/scm_backup_monitor_SDU.json 
   ./tgzmon.py http://dayabay.phys.ntu.edu.tw/data/scm_backup_monitor_C.json
   ./tgzmon.py http://dayabay.ihep.ac.cn/data/scm_backup_monitor_SDU.json http://dayabay.phys.ntu.edu.tw/data/scm_backup_monitor_C.json

   ./tgzmon.py http://dayabay.ihep.ac.cn/data/scm_backup_monitor_SDU.json  -e blyth@hep1.phys.ntu.edu.tw
   ./tgzmon.py http://dayabay.ihep.ac.cn/data/scm_backup_monitor_SDU.json  -e blyth@hep1.phys.ntu.edu.tw,maqm@ihep.ac.cn

Pull JSON data and apply `monitor_` methods. 
When constraint violations are found send notification emails

Usage from cron 

    cd $ENV_HOME/scm ; python- ; ./tgzmon.py -e blyth@hep1.phys.ntu.edu.tw http://dayabay.ihep.ac.cn/data/scm_backup_monitor_SDU.json 


TODO:

#. currently too much TZ noise in this monitoring 

    #. factor this downwards, tis such a common task duplication needs to be avoided


"""
import os, logging
from datetime import datetime, timedelta
log = logging.getLogger(__name__)
from env.plot.highmon import HighMon, CnfMon

import pytz
utc = pytz.utc
dt_utc_ = lambda ts:datetime.utcfromtimestamp(ts).replace(tzinfo=utc)
nw_utc_ = lambda:datetime.utcnow().replace(tzinfo=utc)   
dt_loc_ = lambda dt,tz:dt.astimezone(tz)

def kludge_( series ):
    """
    :param series:
    :return: dt_utc  tz aware datetime instance in UTC 

    The times in the SQLite DB appear to be in CST (naughty naughty boy) hence
    the kludge bringing the timestamp down to UTC to allow that transgression 
    to be sweeped under the carpet. This also allows time handling to follow recommended
    patterns, after the kludge.

    For some good guidelines on python datetime usage, see pytz docs

    * http://pytz.sourceforge.net/

    """
    return series['data'][-1][0]/1000 - 60*60*8            # 8 hrs kludge 

    

class TGZMon(HighMon):
    """
    This `HighMon` subclass collects `monitor_` methods 
    intended to constrain the content of JSON plot data.

    Calling instances of this class, results in the following steps

    #. reads the JSON plot data for each URL
    #. runs all the below `monitor_` methods 
    #. sends single email in case of violations present in any of the URLs

    """
    def __init__(self, cnf ):
        HighMon.__init__(self, cnf )
        self.maxage = timedelta(hours=24)   

    def monitor_val(self, url, method, series ):
        """
        Wide applicability constraint ideas, looking for big changes:
        
        #. deviation of latest value from defined period rolling averages exceeds percentage bounds
        """
        pass

    def monitor_list(self, url, method, series ):
        """
        TODO: sweep some of the mess into functions  
        """
        fmt_ = self.cnf.fmt_
        ts = kludge_(series) 

        dt_utc = dt_utc_(ts)
        dt_loc = dt_loc_(dt_utc,self.cnf.loc)
        msg = " %s  %s " % ( fmt_(dt_utc), fmt_(dt_loc) ) 
        self.add_note( url=url, method=method, series=series['name'], msg=msg )

    def monitor_age(self, url, method, series ):
        """
        :param url: of the JSON data being monitored
        :param method: name 
        :param series: 

        Modulo the TZ shenanigans this monitoring mightbe widely applicable
        to any web accessible HighCharts JSON plot. Motivation to apply a 
        onetime fix to make DB content use UTC.
        """
        ts = kludge_(series) 
        dt_utc = dt_utc_(ts) 
        nw_utc = nw_utc_()
        age = nw_utc - dt_utc                         

        if age > self.maxage:
            msg="age %-25s exceeds maximum allowable %s  " % ( age, self.maxage ) 
            self.add_violation( url=url, method=method, series=series['name'], msg=msg )
        else:
            msg="age %-25s within allowable range %s  " % ( age, self.maxage ) 
            self.add_note( url=url, method=method, series=series['name'], msg=msg )


if __name__ == '__main__':

    durls = [
            "http://dayabay.ihep.ac.cn/data/scm_backup_monitor_SDU.json",
            "http://dayabay.phys.ntu.edu.tw/data/scm_backup_monitor_C.json",
            "http://dayabay.phys.ntu.edu.tw/data/scm_backup_monitor_H1.json",
            ]
    if os.environ.get('NODE_TAG',None) == 'G': 
        durls += ["http://localhost/data/scm_backup_monitor_Z9:229.json"]

    cmon = CnfMon(__doc__)
    cnf = cmon(durls)
    mon = TGZMon(cnf)
    mon()
    print mon
