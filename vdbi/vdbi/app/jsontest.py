

import simplejson as json
import urllib

class JSonGet:
    def __init__(self, baseurl ):
    	self.baseurl = baseurl
    def __call__(self, **kwa ):
    	if not('?' in self.baseurl):
    		url = self.baseurl + '?' + urllib.urlencode(kwa)
    	else:
    		url = self.baseurl
    	return json.load(urllib.urlopen(url))


url = "http://localhost:8080/SimPmtSpecDbis.json?q.ctx.a=and&q.ctx.o=ctx_&q.ctx.c-0.SimFlag=2&q.ctx.c-0.Site=1&q.ctx.c-0.DetectorId=0&q.ctx.c-0.Timestamp=2009-09-11+13%3A33%3A21&q.xtr.a=xtr_&q.xtr.o=and"

def test_basics():
    jsg = JSonGet(url)
    js = jsg()
    assert js.keys() ==  ['items', 'query']
    assert len(js['items']) == 30
    assert js['items'][-1] == {'AD': 1,
     'AFTERPULSE': 0.0149,
     'COLUMN': 1,
     'DARKRATE': 616.89999999999998,
     'EFFIC': 1.3999999999999999,
     'GAIN': 1.04,
     'GFWHM': 0.57599999999999996,
     'PREPULSE': 0.0037299999999999998,
     'RING': 2,
     'ROW': 31,
     'SEQ': 1,
     'SEQNO': 1,
     'SITE': 1,
     'TOFFSET': 53.100000000000001,
     'TSPREAD': 1.95,
     'VAGNO': -1,
     'VEND': '2030-01-01 00:00:00',
     'VINSERT': '2008-10-01 00:00:00',
     'VSIM': 2,
     'VSITE': 1,
     'VSTART': '1980-01-01 00:00:00',
     'VSUB': 0,
     'VTASK': 0,
     'VVERS': '2008-01-01 00:00:00'} 
     
     
     