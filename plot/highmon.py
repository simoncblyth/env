#!/usr/bin/env python
"""
Base class for monitoring JSON data in HighStock format, usage::

    from env.plot.highmon import HighMon

"""
import inspect, urllib2, logging
log = logging.getLogger(__name__)
try:
    import json
except ImportError:
    import simplejson as json
        
def read_json( url ):
    req = urllib2.Request( url )
    opener = urllib2.build_opener()
    f = opener.open(req)
    return json.load(f)


class HighMon(list):
    @classmethod
    def introspect_monitor_methods(cls):
        return [(k,v) for k,v in inspect.getmembers(cls) if k[0:8] == 'monitor_']

    def __init__(self, url):
        self.url = url
        self.js = read_json(url)
        self.mm = self.introspect_monitor_methods()
        list.__init__(self)

    def __call__(self):
        for series in self.js['series']:
            for method_name, method in self.mm:
                method(self, method=method_name, series=series )
  
    def monitor_example_(self, method, series):
        name = series['name']
        data = series['data']
        last = data[-1]
        log.info( "method %s name %s len data %s last %r  " % (method,name,len(data),last ))


class ExampleHighMon(HighMon):
    def __init__(self, url ):
        HighMon.__init__(self, url )
    def monitor_somename(self, method, series ):
        print method
    def monitor_othername(self, method, series ):
        print method



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    url = "http://dayabay.ihep.ac.cn/data/scm_backup_monitor_SDU.json"
    ehm = ExampleHighMon(url)
    ehm()     



