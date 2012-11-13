#!/usr/bin/env python
"""
Base class for monitoring JSON data in HighStock format, usage::

    from env.plot.highmon import HighMon

"""
import os, inspect, urllib2, logging
from env.tools.sendmail import sendmail
log = logging.getLogger(__name__)

try:
    import json
except ImportError:
    import simplejson as json
        
def read_json( url ):
    """
    TODO: handlefailure to connect by returning None
    """
    req = urllib2.Request( url )
    opener = urllib2.build_opener()
    f = opener.open(req)
    return json.load(f)

class Violation(dict):
    def __repr__(self):
        return "method:%(method)s series:%(series)s msg:%(msg)s" % self

class HighMon(list):
    @classmethod
    def introspect_monitor_methods(cls):
        return [(k,v) for k,v in inspect.getmembers(cls) if k[0:8] == 'monitor_']

    def __init__(self, url, email=None):
        """
        Defer loading js to the `__call__`, as that might fail and wish to 
        handle failures by sending notification emails
        """
        self.url = url
        self.email = email
        self.mm = self.introspect_monitor_methods()
        self.js = None
        list.__init__(self)

    def __repr__(self):
        return "\n".join(map(repr, self))

    def add_violation(self, method, series, msg ):
        v = Violation(method=method, series=series, msg=msg )
        self.append(v)

    def _load(self):
        self.js = read_json(self.url)
        if not self.js:
            self.add_violation( method="_load", name="", msg="failed to load JSON from url %s " % self.url )                

    def _constrain(self):
        if not self.js:
            log.warn("skip method calls as json not loaded")
            return
        for series in self.js['series']:
            for method_name, method in self.mm:
                method(self, method=method_name, series=series )

    def _notify(self):
        if len(self) == 0:
            log.info("no violations, not sending email")
            return
        msg = repr(self)
        log.warn("%s violations, sending email\n%s\n" % ( len(self), msg ))
        if self.email:
            for _ in self.email.split():
                log.warn("sendmail to %s " % _ )
                sendmail( msg, _ )
        else:
            log.warn("email address(s) for notification not configured")

    def __call__(self):
        self._load()
        self._constrain()
        self._notify()



class ExampleHighMon(HighMon):
    def __init__(self, url ):
        HighMon.__init__(self, url )
    def monitor_somename(self, method, series ):
        print method
    def monitor_othername(self, method, series ):
        print method
    def monitor_example_(self, method, series):
        name = series['name']
        data = series['data']
        last = data[-1]
        log.info( "method %s name %s len data %s last %r  " % (method,name,len(data),last ))



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    url = "http://dayabay.ihep.ac.cn/data/scm_backup_monitor_SDU.json"
    ehm = ExampleHighMon(url, email=os.environ['SCM_HIGHMON_EMAIL'] )
    ehm()     



