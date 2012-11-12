#!/usr/bin/env python
"""
Standalone checking of json content : to be invoked from cron jobs

In [39]: for series in js['series']:print series['name'], len(series['data']), series['data'][-1]
   ....: 
   OK 76 [1352678400000L, 8]
   svn/dybaux 65 [1352721601000L, 3664.0]
   svn/dybsvn 65 [1352721601000L, 2220.0]
   tracs/dybaux 65 [1352721601000L, 30.0]
   tracs/dybsvn 65 [1352721601000L, 1290.0]

IDEAS:

#. publish a summary json, with just the last month (perhaps) as well as the full thing 

   * not so big currently, but would be more efficient 

"""

import urllib2
try:
    import json
except ImportError:
    import simplejson as json

def read_json( url ):
    req = urllib2.Request( url )
    opener = urllib2.build_opener()
    f = opener.open(req)
    f = opener.open(req)
    return json.load(f)
    
class Monitor(list):
    def __init__(self, js ): 
        self.js = js
        for series in js['series']:
            stamp, value = series['data'][-1] 
            print " %s %s %s %s " % ( series['name'], len(series['data']), stamp, value ) 


if __name__ == '__main__':
    url = "http://dayabay.ihep.ac.cn/data/scm_backup_monitor_SDU.json"
    js = read_json( url )
    mo = Monitor( js)

