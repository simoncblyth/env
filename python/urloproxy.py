#!/usr/bin/env python
"""

   http://www.voidspace.org.uk/python/articles/urllib2.shtml
   http://www.voidspace.org.uk/python/articles/urllib2.shtml#id27

   urllib2 auto-detects http_proxy 

"""




def get(url, proxy):

    print "get url:%s proxy:%s " % ( url , proxy )
    import os
    if proxy:
        os.environ['http_proxy'] = proxy

    import socket
    socket.setdefaulttimeout(5)
    from urllib2 import Request, urlopen, URLError, HTTPError

    req = Request(url)
    try:
        response = urlopen(req)
    except HTTPError, e:
        print e.__class__, e 
    except URLError, e:
        print e.__class__, e 
    else:
        print response
        print response.read()
        # everything is fine


if __name__=='__main__':
    import sys
    print sys.argv[1:]

    sys.exit(get(*sys.argv[1:]))




