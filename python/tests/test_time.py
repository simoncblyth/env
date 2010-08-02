"""
   MDC10b.runLED_Muon.FullChain is coming up with 3600s (1hr wrong) : 
       "Start time in seconds UTC = 1277996400.0"
   due to non-tz/dst aware time handling

   Cleanest way to fix is to move to datetime handling as demoed in check_datetime below


Note: using non-default geometry in $SITEROOT/dybgaudi/Detector/XmlDetDesc/DDDB/dayabay_dry.xml is loaded.
Importing modules MDC10b.runLED_Muon.FullChain [ -M -1 -I 5000 -i 1000 -D 5500. -w 2010-07-01T16:00:00 ]
Trying to call configure() on MDC10b.runLED_Muon.FullChain
Using time format = %Y-%m-%dT%H:%M:%S
Start time in seconds UTC = 1277996400.0

"""
from datetime import tzinfo, timedelta, datetime
import time

ZERO = timedelta(0)
HOUR = timedelta(hours=1)
class UTC(tzinfo):
    """UTC"""
    def utcoffset(self, dt):
        return ZERO
    def tzname(self, dt):
        return "UTC"
    def dst(self, dt):
        return ZERO

utc = UTC()
seconds = 1.278e9
fmt = '%Y-%m-%dT%H:%M:%S'

def test_utc():
    s = seconds 
    d = datetime.fromtimestamp( s, utc )
    print repr(d)
    assert str(d) ==  "2010-07-01 16:00:00+00:00"

def test_fmt():
    t = time.gmtime( seconds )
    f = time.strftime( fmt , t )
    assert f == "2010-07-01T16:00:00"
    tt = time.strptime( f , fmt )      ## this looses tz/dst info 
    assert tt[8] == t[8]  , ("tm_isdst discrepancy", tt[8], t[8] )
    assert tt == t , ( tt, t )
test_fmt.__test__ = False

import os
class Walker(list):
    def __init__(self, root ):
        self.root = root
    def __call__(self): 
        return self.walk_(self.root)
    def walk_(self, base):
        for name in sorted(os.listdir(base)):
            path = os.path.join(base, name) 
            isdir = os.path.isdir(path) 
            if isdir:
                for x in self.walk_(path):
                    yield x       
            if not(path.endswith('.tab')):
                yield path[len(self.root)+1:]


tzs = Walker("/usr/share/zoneinfo")()

def check_time( tz ):
    """
         http://www.tutorialspoint.com/python/python_date_time.htm
    
           time.gmtime(epoch)
               Accepts an instant expressed in seconds since the epoch and returns 
               a time-tuple t with the UTC time. Note : t.tm_isdst is always 0 

           time.mktime(tuple) -> floating point number
               Accepts an instant expressed as a time-tuple in local time and returns a 
               floating-point value with the instant expressed in seconds since the epoch.

           Fails in 525 out of 589 timezones due to not handling DST ... avoid the 
           hassles of correction by using datetime instead

    """
    os.environ['TZ'] = tz
    s = seconds
    f = fmt
    t = time.gmtime( s )                    # construct UTC time from epoch seconds
    d = time.strftime( f , t )              # format ... without tz/dst info
    p = time.strptime( d , f )              # parse the formatted ... loosing tz/dst info
    c = time.mktime( p ) - time.timezone
    print locals() 
    assert s == c , locals()
 
def check_datetime( tz ):
    """
         Use the relatively sane datetime 
         ... succeeds in 589 timezones to roundtrip from the epoch seconds 
         into a formatted string then parse back into datetime and thence back to epoch seconds

    """
    os.environ['TZ'] = tz
    s = seconds
    t = datetime.fromtimestamp( s, utc )           # construct datetime explicitly in UTC
    t0 = datetime.fromtimestamp( 0 , utc )         # start of the epoch 
    f = fmt + '-%Z' 
    d = t.strftime( f )                            # format ... with tz appended BUT tiz pointless as parsing will not see it
    p = datetime.strptime( d , f )                 # parse the formatted ... loosing tz/dst info
    p = p.replace( tzinfo = utc )                  # replace lost tzinfo  
    
    assert p == t , ( "mismatch between orig and parsed ", locals() )
    
    dt = p - t0                       
    c  = dt.seconds + dt.days * 24 * 3600 

    print locals()
    assert s == c , locals()
 

def test_time():
    for tz in tzs:
        yield check_time, tz
test_time.__test__ = False

def test_datetime():
    for tz in tzs:
        yield check_datetime, tz
test_datetime.__test__ = True

if __name__=='__main__':
    pass
    #test_utc()
    #test_fmt()
    for tz in tzs:
        #check_time( tz )
        check_datetime( tz )


