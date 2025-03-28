#!/usr/bin/env python
"""
Base class for monitoring JSON data in HighStock format, usage::

    from env.plot.highmon import HighMon

Lots of what this is doing follows nosetesting. 
Perhaps nose could be used here in an indirect manner.

"""
import os, inspect, urllib2, logging, pytz
from datetime import datetime
from env.tools.sendmail import sendmail
log = logging.getLogger(__name__)
fmt_ = lambda _:_.strftime('%Y-%m-%d %H:%M:%S %Z%z')

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
    try:
        f = opener.open(req)
    except urllib2.HTTPError:
        log.warn("failed to load from url %s " % url )
        f = None
    return json.load(f) if f else None

class Violation(dict):
    def __repr__(self):
        return "FAIL : %(url)s %(method)s %(series)-30s %(msg)s" % self
class Note(dict):
    def __repr__(self):
        return "PASS : %(method)s %(series)-30s %(msg)s" % self
class Anno(dict):
     """
     URL keyed dict-of-lists of notes : allowing notes presentation per-URL
     """
     def __init__(self, urls, type="Violations" ):
         dict.__init__(self)
         self.urls = urls
         self.type = type
         for url in urls:
             self[url] = []
     def __repr__(self):
          ret = []
          for url in self.urls:
              nvi = len(self[url]) 
              if nvi == 0:continue
              ret.append("")
              ret.append(url + " %s:%s" % (self.type, nvi) )
              ret.append("")
              ret.append( "\n".join( map(repr,self[url]) ))
          return "\n".join(ret)
     def ucountd(self):
         """
         :return: url keyed dict with list count values
         """
         return dict(map(lambda url:(url,len(self[url])),self.urls))
     def ucounts(self):
         """
         :return: list of counts in url order
         """
         return map(lambda url:len(self[url]), self.urls)
     def _count(self):
         ucounts = self.ucounts()
         return sum(ucounts)
     count = property(_count)


class CnfMon(object):
    def __init__(self, doc ):
        """
        :param doc:  docstring

        Note using OptionParser rather than argparser for older python compatibility, but 
        doing things in an argparser manner.

        The setup and parsing is split to allow the addition of options from callers.
        """
        from optparse import OptionParser
        parser = OptionParser(usage=doc)
        parser.add_option("-l", "--level", default="INFO", help="logging level")
        parser.add_option("-z", "--timezone", default="Asia/Taipei", help="pytz timezone string used for localtime outputs ")
        parser.add_option("-e", "--email",    default=os.environ.get('MAILTO', None), help="Comma delimited email addresses for notification")
        parser.add_option(      "--reminder", default="Tue",  help="Abbreviated day of week `strtime %a` on which to send reminder email, even when no violations.")
        self.parser = parser

    def __call__(self, urls_):
        """
        :param urls_: default JSON urls
        :return: OptionParser instance, with some tack-ons `.loc` and `.urls` `.fmt_`
        """
        opts, args = self.parser.parse_args()
        logging.basicConfig(level=getattr(logging, opts.level.upper()))
        opts.loc = pytz.timezone(opts.timezone)   
        opts.urls = urls_ if len(args) == 0 else args    # poor-mans argparser
        opts.fmt_ = lambda _:_.strftime('%Y-%m-%d %H:%M:%S %Z%z')
        return opts


class HighMon(object):
    @classmethod
    def introspect_monitor_methods(cls):
        return [(k,v) for k,v in inspect.getmembers(cls) if k[0:8] == 'monitor_']

    def __init__(self, cnf):
        """
        :param cnf: Instance of `CnfMon` including properties

        `urls`
                list of urls from which to pull
        `email`
                comma delimited string with notification email addresses
        `reminder`
                abbreviated day of week on which to send notifications, even when no problems
                as weekly reassurance that the cron task is active 

        Defer loading js to the `__call__`, as that might fail and wish to 
        handle failures by sending notification emails
        """

        self.cnf = cnf
        urls = list(cnf.urls)
        self.urls = urls
        self.email = cnf.email
        self.notes = Anno(urls, "notes")
        self.violations = Anno(urls, "violations")
        today = datetime.utcnow().strftime("%a") 
        self.reminder = cnf.reminder if today == cnf.reminder else ""
        self.mm = self.introspect_monitor_methods()
        self.js = {}

    def hdr(self):
        return "%s [%s] %s URLs %s Violations per URL : %r " % ( self.__class__.__name__ , self.reminder, len(self.urls) , fmt_(datetime.now()), self.violations.ucounts() )

    def __repr__(self):
        return "\n".join([self.hdr()]+[repr(self.violations)]+[repr(self.notes)])

    def add_violation(self, url, method, series, msg ):
        v = Violation(url=url,method=method, series=series, msg=msg )
        self.violations[url].append(v)

    def add_note(self, url, method, series, msg ):
        n = Note(url=url,method=method, series=series, msg=msg )
        self.notes[url].append(n)

    def _load(self, url):
        log.info("_load json from %s " % url )
        self.js[url] = read_json(url)
        if not self.js[url]:
            self.add_violation( url=url, method="_load", series="", msg="failed to load JSON from url %s " % url )                
        else:
            self.add_note( url=url, method="_load", series="", msg="succeeded to load JSON " )                

    def _constrain(self, url):
        if not self.js[url]:
            log.warn("skip method calls failed to load json from url %s " % url )
            return
        for method_name, method in self.mm:
            for series in self.js[url]['series']:
                method(self, url=url, method=method_name, series=series )

    def _notify(self):
        if not self.reminder and self.violations.count == 0:
            log.info("no violations, not sending email" )
            return
        msg = repr(self)
        log.warn("%s violations, reminder? [%s],  sending email\n%s\n" % ( self.violations.count, self.reminder, msg ))
        if self.email:
            for _ in self.email.split(","):
                log.warn("sendmail to %s " % _ )
                sendmail( msg, _ )
        else:
            log.warn("email address(s) for notification not configured")

    def __call__(self):
        for url in self.urls:
            self._load(url)
            self._constrain(url)
            pass
        self._notify()



class ExampleHighMon(HighMon):
    def __init__(self, cnf  ):
        HighMon.__init__(self, cnf )
    def monitor_somename(self, url, method, series ):
        pass
    def monitor_othername(self, url, method, series ):
        pass
    def monitor_example_(self, url, method, series):
        """
        .. warn:: no violations added here
        """
        name = series['name']
        data = series['data']
        last = data[-1]
        ts = data[-1][0]/1000 - 60*60*8
        dt =  datetime.fromtimestamp(ts)
        log.info( "method %s name %-30s len data %s last %-40s %s " % (method,name,len(data),last, fmt_(dt) ))



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    class Cnf(object):
        email = os.environ.get('MAILTO',None) 
        reminder = 'Wed'
        urls = (
            "http://dayabay.ihep.ac.cn/data/scm_backup_monitor_SDU.json",
            "http://dayabay.phys.ntu.edu.tw/data/scm_backup_monitor_C.json",
            "http://dayabay.phys.ntu.edu.tw/data/scm_backup_monitor_H1.json",
            "http://localhost/data/scm_backup_monitor_Z9:229.json",
            )

    cnf = Cnf()    
    ehm = ExampleHighMon(cnf)
    ehm()     


