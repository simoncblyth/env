

from rum.query import *
from formencode import variabledecode

from vdbi import debug_here
from vdbi.dyb import ctx

import logging
log = logging.getLogger(__name__)


ctx_complete = lambda v:'SimFlag' in v and 'Site' in v and 'DetectorId' in v and 'Timestamp' in v


def ctx_expression(v):
    if ctx_complete(v):
        return and_([ 
           eq(ctx['Site.attr'],v['Site']) ,
           eq(ctx['SimFlag.attr'],v['SimFlag']) ,
           eq(ctx['DetectorId.attr'],v['DetectorId']) ,    
           lt(ctx['TimeStart.attr'],v['Timestamp']),
           gt(ctx['TimeEnd.attr'],v['Timestamp']),
                 ])
    return None



class ReContext(dict):
    def __init__(self, d):
        self.d = d
        log.debug("ReContext.__init__ %s " % repr(d) )
        if 'q' in d and 'c' in d['q']:
            self.recon_ctx_(d['q'])
    def recon_ctx_(self, d):
        """
           recurse picking up the context variables ..
           hmmm perhaps should only take when complete context is obtained
           from the same level
        """
        if 'c' in d:
            if isinstance(d['c'], (list,tuple)):
                for i in d['c']:
                    self.recon_ctx_(i)
            elif isinstance(d['c'],str):
                if d.get('a',None) and d.get('o',None):
                    if ctx.get(d['c'],None):
                        self[ctx[d['c']]] = d['a']
    
    def __call__(self):
        d = self.d
        if ctx_complete(self):
            d['ctx'] = {}
            d['ctx']['c'] = [dict(self)]
            d['ctx']['a'] = None
            d['ctx']['o'] = "and"
        else:
            log.debug("incomplete context %s " % repr(self) )
        log.debug("ReContext.__call__ %s " % repr(d) )
        return d



class DbiQueryFactory(QueryFactory):
 
    def __call__(self, resource, request_args=None, **kw):
         """
              turns the request into a Query
              with DBI context criteria added  
             
         """
         query = super(DbiQueryFactory, self).__call__(resource, request_args=request_args, **kw ) 
         d = {}
         if request_args:
             d = variabledecode.variable_decode(request_args)
         
         log.debug("DbiQueryFactory.__call__ d %s " % repr(d) ) 
         
         if 'ctx' in d and 'c' in d['ctx']:
             vs = d['ctx']['c']
             if isinstance(vs, (list,tuple) ) and len(vs) > 0:
                 v = vs[0]       ## only the 1st value 
                 expr =  ctx_expression(v)
                 if not(expr):
                     log.debug("invalid ctx query ")                
                 else:
                     if query.expr:
                         query = query.clone( expr = and_( [query.expr, expr ]) ) 
                         log.debug("combined query %s" % repr(query) )  
                     else:
                         query = query.clone( expr = expr ) 
                         log.debug("ctx only query %s" % repr(query) ) 
             else:
                 log.error("incomplete ctx query dict %s " % repr(v) )
         else:
             log.debug("no ctx query")
         
                 
         return query 


if __name__=='__main__':
    q = Query(and_([eq('VSITE', u'1'), eq('VSIM', u'2'), eq('VSUB', u'0'), lt('VSTART', u'2009/09/01 15:17'), gt('VEND', u'2009/09/01 15:17')]), None, 30, None)
    d = q.as_dict()
    print d
    d2 = ReContext(d)()
    print d2
    
    expr = and_([eq('VSITE',1)])
    qq = q.clone( expr = and_([q.expr, expr]))
    dd = qq.as_dict()
    dd2 = ReContext(dd)()
    
    
 
    