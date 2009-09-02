

from rum.query import *
from formencode import variabledecode
from formencode.validators import Int

from vdbi import debug_here
from vdbi.dyb import ctx

import logging
log = logging.getLogger(__name__)






class ReContext(dict):
    def __init__(self, d):
        """
           Used in the DbiQueryBuilder.adapt_value , which converts
           a Query object into the dict needed to fill-out the values in 
           the widgets ... due to the changed widget layout the dict 
           returned from Query.as_dict needs correction to fit 
        
        """
        self.d = d
        log.debug("ReContext.__init__ %s " % repr(d) )
        if d and 'q' in d and 'c' in d['q']:     ## standard old dict layout 
            self.recon_ctx_(d['q'])
            
    def recon_ctx_(self, d):
        """
           recurse picking up the context variables ..
           hmmm perhaps should only take when complete context is obtained
           from the same level
           
           TODO : pop the values obtained to prevent duplication in the interface
           the context parameters currently leak in the the extras  
           
            d['q']['xtr']['c'].pop(2)
           
           
        """
        if 'c' in d:
            if isinstance(d['c'], (list,tuple)):
                dc = d['c']
                for i,dci in enumerate(dc):
                    if self.recon_ctx_(dci):
                        d['c'].pop(i)
            elif isinstance(d['c'],str):
                if d.get('a',None) and d.get('o',None):
                    if ctx.get(d['c'],None):
                        a = d['a']
                        try:
                            ia = int(a)
                        except ValueError:
                            ia = a
                        self[ctx[d['c']]] = ia
                        return True  ## signal a pop
        return False        
    
    def __call__(self):
        d = self.d
        
        n = {}
        n['q'] = {}
        if d and 'q' in d:
            n['q']['xtr'] = d['q']
        
        if ctx_complete(self):
            n['q']['ctx'] = {}
            n['q']['ctx']['c'] = [dict(self)]
            n['q']['ctx']['o'] = u"and"
        else:
            log.debug("incomplete context %s " % repr(self) )
        log.debug("ReContext.__call__ %s " % repr(n) )
        return n



 



ctx_complete = lambda v:'SimFlag' in v and 'Site' in v and 'DetectorId' in v and 'Timestamp' in v

def ctx_expression(d):
    if 'c' in d:
        vs = d['c']
        if isinstance(vs, (list,tuple) ) and len(vs) > 0:
            v = vs[0]       ## only the 1st value 
            if ctx_complete(v):
                return and_([ 
                   eq(ctx['Site.attr'],v['Site']) ,
                   eq(ctx['SimFlag.attr'],v['SimFlag']) ,
                   eq(ctx['DetectorId.attr'],v['DetectorId']) ,    
                   lt(ctx['TimeStart.attr'],v['Timestamp']),
                   gt(ctx['TimeEnd.attr'],v['Timestamp']),
                         ])
    return None


class DbiQueryFactory(QueryFactory):
 
    def __call__(self, resource, request_args=None, **kw):
         """
      Convert the request into a Query with DBI context criteria added  
            
      The components of the Query are prepared here (rather than in Query.from_dict)
      and feed into the QueryFactory for rubberstamping into a Query object 
      ( done via the kw rather than the request_args )
                
      This needs to be done due to the changed widget layout     
             
         """
         log.debug("DbiQueryFactory.__call__ req = %s " % repr(request_args) ) 
         
         if request_args:
             d = variabledecode.variable_decode(request_args)
             expr = sort = limit = offset = None
             
             xtr_expr = None
             if 'q' in d and 'xtr' in d['q'] and 'c' in d['q']['xtr']:
                 xtr_expr = Expression.from_dict(d['q']['xtr'])
                 log.debug("xtr_expr %s " % xtr_expr )     
                  
             ctx_expr = None
             if 'q' in d and 'ctx' in d['q'] and 'c' in d['q']['ctx']:
                 ctx_expr = ctx_expression( d['q']['ctx'] )  
                 log.debug("ctx_expr %s " % ctx_expr )        
                 if not(ctx_expr):
                     log.debug("invalid ctx query %s " % d['q']['ctx'] )    
                     
             if xtr_expr and ctx_expr:
                 expr = and_( [xtr_expr, ctx_expr ])
             elif xtr_expr:
                 expr = xtr_expr
             else:
                 expr = ctx_expr
                              
             if 'sort' in d:
                 sort = d['sort']
                 if sort:sort = map(_sort.from_dict, sort)
                 
             if 'limit' in d:
                 limit = Int(min=0).to_python(d['limit'])
                 
             if 'offset' in d:
                 offset = Int(min=0).to_python(d['offset'])

             kw = { 'expr':expr , 'sort':sort , 'limit':limit , 'offset':offset }
             request_args = None   ## KILL THE REQUEST ... 
        
        
         query = super(DbiQueryFactory, self).__call__(resource, request_args=request_args , **kw ) 
         log.debug("DbiQueryFactory ... query %s " % query ) 
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
    
    
    from webob.multidict import MultiDict, UnicodeMultiDict
    multi = MultiDict([('q.ctx.o', u'and'), ('q.ctx.c-0.SimFlag', u'2'), ('q.ctx.c-0.Site', u'1'), ('q.ctx.c-0.DetectorId', u'0'), ('q.ctx.c-0.Timestamp', u'2009/09/02 18:34'), ('q.xtr.o', u'and')]) 
    req = UnicodeMultiDict( multi )
    d = variabledecode.variable_decode(req)
    
    
    