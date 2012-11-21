#!/usr/bin/env python
"""

.. warn:: **no longer used by SCM machinery** : retain for when cannot use pytz

Demo::

    In [102]: utcnow = datetime.utcnow().replace(tzinfo=utc)  
    In [103]: utcnow

    Out[103]: datetime.datetime(2012, 11, 13, 8, 9, 36, 158505, tzinfo=<timezone.UTC object at 0x13d22b0>)

    In [104]: print utcnow
    2012-11-13 08:09:36.158505+00:00

    In [106]: print utcnow.strftime("%c")
    Tue Nov 13 08:09:36 2012

    In [108]: print utcnow.astimezone(cst)    # explicitly express now in local time
    2012-11-13 16:09:36.158505+08:00



 * http://www.enricozini.org/2009/debian/using-python-datetime/
 * http://pytz.sourceforge.net/

"""
from datetime import tzinfo, timedelta, datetime
ZERO = timedelta(0)
EIGHT = timedelta(hours=8)

class UTC(tzinfo):
    """UTC"""
    def utcoffset(self, dt):
        return ZERO
    def tzname(self, dt):
        return "UTC"
    def dst(self, dt):
        return ZERO

class CST(tzinfo):
    """CST"""
    def utcoffset(self, dt):
        return EIGHT
    def tzname(self, dt):
        return "CST"
    def dst(self, dt):
        return ZERO

utc = UTC()
cst = CST()

if __name__ == '__main__':
    from pprint import pformat
    ts = 1342102203000/1000
    d = {}
    d['local'] = datetime.fromtimestamp(ts)
    d['utc'] = datetime.fromtimestamp(ts, utc)
    d['cst'] = datetime.fromtimestamp(ts, cst)

    utcnow = datetime.utcnow().replace(tzinfo=utc)      
    print utcnow    
    print utcnow.astimezone(cst)
    print pformat(d) 


