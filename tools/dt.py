
import re
import time 
from datetime import datetime

def parsetime(ds,fmt):
    if hasattr(datetime, 'strptime'):
         return datetime.strptime(ds,fmt)
    else:
         return datetime(*(time.strptime(ds, fmt)[0:6])) 

class DT(str):
    _ptn = re.compile("(?P<parsable>\d\d\d\d-\d\d-\d\dT\d\d:\d\d:\d\d)(?P<extra>\.\d*Z)")
    _fmt = "%Y-%m-%dT%H:%M:%S"
    parsable = property(lambda self:self._ptn.match(self).groupdict()['parsable'] )
    t = property( lambda self:parsetime( self.parsable, self._fmt ))

if __name__ == '__main__':
    d = DT("2011-03-07T03:00:09.061005Z")
    print d.t




