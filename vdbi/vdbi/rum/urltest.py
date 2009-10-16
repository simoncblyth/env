
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
    
    def od_kludged(self):
        """
            Fixup the original widget dict to match the one 
            that has traversed thru the Query 
        """
        od = self.od
        if not(od['q']['plt'].get('a',None)):
            od['q']['plt'].update( { 'a':{'param': {} } })  ## add empty plt param for agreement 
        return od
    
    req   = property(lambda self:UnicodeMultiDict( self.raw.GET ))
    od    = property(lambda self:variabledecode.variable_decode(self.req))
    odk   = property(od_kludged)
    comps = property(lambda self:_vdbi_comps(self.od))
    dq    = property(lambda self:_vdbi_expression(self.od)['q'] )
    expr  = property(lambda self:Expression.from_dict( self.dq ))
    q     = property(lambda self:Query.from_dict( self.req ))
    qd    = property(lambda self:self.q.as_dict())
    qdw   = property(lambda self:_vdbi_widget(self.qd))
    qfd   = property(lambda self:self.q.as_flat_dict())
    qfdw  = property(lambda self:self.q.as_flat_dict_for_widgets())
    rurl  = property(lambda self:self.raw.path_url + "?" + "&".join(["%s=%s" % ( k,v) for k,v in self.qfdw.items()]) )
    repl  = property(lambda self:URLtest(self.rurl))

    def checks(self):
        self.qexpr_consistent()
        self.passage_thru_query()

    def qexpr_consistent(self):
        assert `self.expr` == `self.q.expr` 
        
    def passage_thru_query(self):
        odk = self.odk
        qdw = self.qdw
        assert qdw == odk , "passage_thru_query fails \n%s\n %s " % ( `odk` , `qdw`)



if __name__ == '__main__':
    u = URLtest.pbpaste()
    assert u.raw.url == u.rurl , "reconstructed url does not match original "
    

