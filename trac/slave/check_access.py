"""
Usage::

  python check_access.py http://dayabay.ihep.ac.cn/tracs/dybsvn dayabay youknowit

"""
import sys
import urllib2

class SaneHTTPErrorProcessor(urllib2.HTTPErrorProcessor):pass

class SaneHTTPRequest(urllib2.Request):
    def __init__(self, method, url, data=None, headers={}):
        urllib2.Request.__init__(self, url, data, headers)
        self.method = method

    def get_method(self):
        if self.method is None:
            self.method = self.has_data() and 'POST' or 'GET'
        return self.method


def check_access( url, username, password , method='POST' ):
    password_mgr = urllib2.HTTPPasswordMgrWithDefaultRealm()
    password_mgr.add_password(None, url, username, password)
    opener = urllib2.build_opener(SaneHTTPErrorProcessor)
    opener.add_handler(urllib2.HTTPBasicAuthHandler(password_mgr))
    opener.add_handler(urllib2.HTTPDigestAuthHandler(password_mgr))

    xml = "<testpost/>"
    body = str(xml)
    headers = {
              'Content-Length': len(body),
              'Content-Type': 'application/x-bitten+xml'
         }
    req = urllib2.Request( url, body, headers or {})

    try:
        return opener.open(req)
    except urllib2.HTTPError, e:
        print 'Server returned error %d: %s' % (e.code, e.msg)


if __name__ == '__main__':
    assert len(sys.argv) == 4 , ("Require 3 args : url username password ", sys.argv )
    print check_access(*sys.argv[1:]).read()
 

