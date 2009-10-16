
from vdbi.rum.query import *
from vdbi.rum.query import _vdbi_expression, _vdbi_widget, _vdbi_comps
from webob import Request, UnicodeMultiDict

class URLtest(object):
    """    
            od : original dict expected to be in widget form
         comps : component branches, used to construct the expression dict
            dq : place the comps in the expression dict 
          expr : expression constructed  from the expression dict
             q : query from the expression + the rest 
            qd : query as_dict
           qdw : query as_dict converted to widget form 
           
           qfd : query as_flat_dict (uses as_dict)                  
          rurl : reconstruced url from the roundtripped url -> query -> qfd
    
    """
    @classmethod
    def pbpaste(cls):
        """
           OSX only, after (cmd-C) copying the desired URL to study         
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
    dq    = property(lambda self:_vdbi_expression(self.od)['q'] )
    expr  = property(lambda self:Expression.from_dict( self.dq ))
    q     = property(lambda self:Query.from_dict( self.req ))
    qd    = property(lambda self:self.q.as_dict())
    qdw   = property(lambda self:_vdbi_widget(self.qd))
    qfd   = property(lambda self:self.q.as_flat_dict())
    rurl  = property(lambda self:self.raw.path_url + "?" + "&".join(["%s=%s" % ( k,v) for k,v in self.qfd.items()]) )
    repl  = property(lambda self:URLtest(self.rurl))

    def consistent(self):
        assert `self.expr` == `self.q.expr` 
        assert self.qdw == self.od
        


if __name__ == '__main__':
    u = URLtest.pbpaste()
    assert u.raw.url == u.rurl , "reconstructed url does not match original "
    

