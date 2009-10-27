import urllib2
import urllib
import simplejson 

from private import Private
p = Private()
creds = urllib.urlencode({ 'username':p('DAYABAY_USER'), 'password':p('DAYABAY_PASS'), } )

def grab( url ):
    req = urllib2.Request( url , creds )
    return simplejson.load( urllib2.urlopen(req) )

if __name__=='__main__':
    import sys
    if len(sys.argv) < 2:
        url = 'http://localhost/dbi/SimPmtSpecDbis.json?limit=1&offset=10'
    else:
        url = sys.argv[1]
    d = grab(url)
    assert 'items' in d.keys() and 'query' in d.keys()
    for i in d['items']:
        print "%s %s " % ( i.get('ROW','-') , repr(i) ) 



