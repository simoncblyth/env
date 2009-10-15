
from vdbi.rum.query import *
from vdbi.rum.query import _vdbi_expression, _vdbi_widget, _vdbi_comps

from webob import Request, UnicodeMultiDict

class URLtest(object):
    """    
         od : original dict expected to be in widget form
    """
    @classmethod
    def pbpaste(cls):
        """
           OSX only, after (cmd-C) copying the desired URL to study         
                    u = URLtest.pbpaste()
        
        """
        import os
        url = os.popen("pbpaste").read()
        if url.startswith('http'):
            return URLtest(url)
        else:
            print "pasted test not a url \"%s\"" % url
        return None
    
    def __init__(self, url):
        if issubclass(url.__class__, Request):
            self.raw = url
        else:
            self.raw = Request.blank( url )  
    
    req   = property(lambda self:UnicodeMultiDict( self.raw.GET ))
    od    = property(lambda self:variabledecode.variable_decode(self.req))
    comps = property(lambda self:_vdbi_comps(self.od))
    exprd = property(lambda self:_vdbi_expression(self.od))
    expr  = property(lambda self:Expression.from_dict( self.exprd['q'] ))
    q     = property(lambda self:Query.from_dict( self.req ))
    qd    = property(lambda self:self.q.as_dict())
    qfd   = property(lambda self:self.q.as_flat_dict())


URLtest.rurl = property(lambda self:self.raw.path_url + "?" + "&".join(["%s=%s" % ( k,v) for k,v in self.qfd.items()]) ))


if __name__ == '__main__':
    u = URLtest.pbpaste()
    assert u.raw.url == u.rurl , "reconstructed url does not match original "
    

