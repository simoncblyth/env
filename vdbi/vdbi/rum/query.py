

from rum.query import *
from rum.query import _sort
from formencode import variabledecode
from formencode.validators import Int, Invalid

from vdbi import debug_here
from vdbi.dyb import ctx

import logging
log = logging.getLogger(__name__)



class ReContext(dict):
    def __init__(self, a):
        """
           Used in the DbiQueryBuilder.adapt_value , which converts
           a Query object into the dict needed to fill-out the values in 
           the widgets ... due to the changed widget layout the dict 
           returned from Query.as_dict needs correction to fit 
        
        """
        from copy import deepcopy
        d = deepcopy(a)
        self.d = d
        
        self['ctx'] = {}
        self['xtr'] = []
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
           
           
          Popping while iterating is problematic  
           
        """
        #print "recon_ctx_ %s " % (repr(d))
        if 'c' in d:
            if isinstance(d['c'], (list,tuple)):
                dc = d['c']
                poplist = []
                
                
                
                for i,dci in enumerate(dc):
                    if self.recon_ctx_(dci):poplist.append(i)
                for i in reversed(poplist):
                    d['c'].pop(i)
                    
            elif isinstance(d['c'],(str,unicode,)):
                if d.has_key('a') and d.has_key('o') and not( d.get('a',None) == None ):
                    try:
                        ia = Int(min=0).to_python(d['a'])
                    except Invalid:
                        ia = d['a']   
                                        
                    ## divide columns into ctx and xtr    
                    a2n = ctx['_attr2name']        
                    if a2n.get(d['c'],None):
                        self['ctx'][ a2n[d['c']] ]  = ia
                    else:
                        self['xtr'].append( {'a':ia , 'o':d.get('o',None) , 'c':d.get('c',None)   } )
                    return True  ## signal a pop
                        
            else:
                log.debug("dc not list or string %s " % repr(d['c']) )
        else:
            log.debug("no c in d ")
        return False        
    
    def __call__(self):
        d = self.d
        
        n = {}
        n['q'] = {}
        if d and 'q' in d:
            n['q']['xtr'] = {}
            n['q']['xtr']['c'] = list(self['xtr'])
            
            ## this is the outer operator that should always be and ...
            ## need to access the inner one that abuts with the extras 
            
            n['q']['xtr']['o'] = d['q'].get('o', None)
        
        if ctx_complete(self['ctx']):
            n['q']['ctx'] = {}
            n['q']['ctx']['c'] = [dict(self['ctx'])]
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
                n2a = ctx['_name2attr']
                return and_([ 
                   eq(n2a['Site'],v['Site']) ,
                   eq(n2a['SimFlag'],v['SimFlag']) ,
                   eq(n2a['DetectorId'],v['DetectorId']) ,    
                   lt(n2a['TimeStart'],v['Timestamp']),
                   gt(n2a['TimeEnd'],v['Timestamp']),
                         ])
    return None


def req2expr(d):
    """
        this 
    """
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
    elif ctx_expr:
        expr = ctx_expr
    else:
        log.error("req2expr neither ctx or xtr found ")
        expr = None
        
    return expr



class DbiQueryFactory(QueryFactory):
 
    def __call__(self, resource, request_args=None, **kw):
         """
      Convert the request into a Query with DBI context criteria added  ... this 
      query is then fed to the widgets for display 
            
      The components of the Query are prepared here (rather than in Query.from_dict)
      and feed into the QueryFactory for rubberstamping into a Query object 
      ( done via the kw rather than the request_args )
                
      This needs to be done due to the changed widget layout     
             
         """
         log.debug("DbiQueryFactory.__call__ req = %s " % repr(request_args) ) 
         
         if request_args:
             d = variabledecode.variable_decode(request_args)
             expr = sort = limit = offset = None
        
             if 'q' in d and not( 'ctx' in d['q'] or 'xtr' in d['q'] ):
                 log.debug( "ReContexting the request ")
                 d = ReContext(d)()    
        
             expr = req2expr(d)
                              
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

 
  
  
def test_ctxonly():
    q = Query(and_([eq('VSITE', u'1'), eq('VSIM', u'2'), eq('VSUB', u'0'), lt('VSTART', u'2009/09/01 15:17'), gt('VEND', u'2009/09/01 15:17')]), None, 30, None)
    d = q.as_dict()
    
    assert d == {'limit': 30,
     'q': {'a': None,
           'c': [{'a': '1', 'c': 'VSITE', 'o': 'eq'},
                 {'a': '2', 'c': 'VSIM', 'o': 'eq'},
                 {'a': '0', 'c': 'VSUB', 'o': 'eq'},
                 {'a': '2009/09/01 15:17', 'c': 'VSTART', 'o': 'lt'},
                 {'a': '2009/09/01 15:17', 'c': 'VEND', 'o': 'gt'}],
           'o': 'and'}} , "raw dict obtained from query mismatch  %s " % repr(d)
    
    r = ReContext(d)
    assert dict(r) == {'ctx': {'DetectorId': 0,
                               'SimFlag': 2,
                                  'Site': 1,
                             'Timestamp': '2009/09/01 15:17'},
                       'xtr': []} , "ReContext extract mismatch %s " % repr(r)
    
    d2 = r()
    assert d2 ==  {'q': 
                     {
                    'ctx': {'c': [{'DetectorId': 0, 'SimFlag': 2, 'Site': 1,'Timestamp': '2009/09/01 15:17'}],'o': u'and'},
                    'xtr': {'c': [], 'o': u'and'}
                     }
                  } , "reconstructed dict mismatch %s " % (repr(d2))
    
    

def test_ctx_and_xtr():
    q = Query(and_([eq('VSITE', u'1'), eq('VSIM', u'2'), eq('VSUB', u'0'), lt('VSTART', u'2009/09/01 15:17'), gt('VEND', u'2009/09/01 15:17')]), None, 30, None)
    
    assert q.as_dict() == {'limit': 30,
                               'q': {'a': None,
                                     'c': [{'a': '1', 'c': 'VSITE', 'o': 'eq'},
                                           {'a': '2', 'c': 'VSIM', 'o': 'eq'},
                                           {'a': '0', 'c': 'VSUB', 'o': 'eq'},
                                           {'a': '2009/09/01 15:17', 'c': 'VSTART', 'o': 'lt'},
                                           {'a': '2009/09/01 15:17', 'c': 'VEND', 'o': 'gt'}],
                                     'o': 'and'}
                                     }
    
    xtr = and_([eq('RING',u'2')])
    qq = q.clone( expr = and_([q.expr, xtr]))
    d = qq.as_dict()
    r = ReContext(d)
    
    assert dict(r) == { 'ctx': {'DetectorId': 0,
                                   'SimFlag': 2,
                                      'Site': 1,
                                 'Timestamp': '2009/09/01 15:17'},
                        'xtr': [{'a': 2, 'c': 'RING', 'o': 'eq'}]}
    
    d2 = r()
    assert d2 == {'q': {'ctx': {'c': [{'DetectorId': 0, 'SimFlag': 2, 'Site': 1, 'Timestamp': '2009/09/01 15:17'}], 'o': u'and'},
                        'xtr': {'c': [{'a': 2, 'c': 'RING', 'o': 'eq'}], 'o': u'and'}}}
    




def test_ctxquery_with_xtr():
    """
       the form of the request URL and the resulting dict that it is 
       decoded into reflects the widget layout 
       
       NB the expr does not currently use the ctx_ ... but is in a form that 
          can be converted into the db select 
    """
    from webob import Request, UnicodeMultiDict
    raw = Request.blank("/SimPmtSpecDbis?q.ctx.o=and&q.ctx.c-0.SimFlag=2&q.ctx.c-0.Site=1&q.ctx.c-0.DetectorId=0&q.ctx.c-0.Timestamp=2009%2F09%2F03+17%3A53&q.xtr.o=and&q.xtr.c-0.c=RING&q.xtr.c-0.o=eq&q.xtr.c-0.a=4")
    req = UnicodeMultiDict( raw.GET )
    from formencode import variabledecode
    d =  variabledecode.variable_decode(req)
    
    orig = {'q': {'ctx': {'c': [{'DetectorId': u"0",
                          'SimFlag': u"2",
                          'Site': u"1",
                          'Timestamp': u'2009/09/03 17:53'}],
                   'o': u'and'},
           'xtr': {'c': [{'a': u'4', 'c': u'RING', 'o': u'eq'}], 'o': u'and'}}}
    
    assert d == orig 
    expr = req2expr( d )
    
    assert expr ==  and_([and_([eq(u'RING', u'4')]), and_([eq('VSITE', u'1'), eq('VSIM', u'2'), eq('VSUB', u'0'), lt('VSTART', u'2009/09/03 17:53'), gt('VEND', u'2009/09/03 17:53')])]) 
    print "expr %s " % expr  
      
    qq = Query( expr=expr )
    dd = qq.as_dict()
    assert dd == {'q': 
                       {'a': None,
                        'c': [{'a': None,
                               'c': [{'a': '4', 'c': 'RING', 'o': 'eq'}],
                               'o': 'and'},
                              {'a': None,
                               'c': [{'a': '1', 'c': 'VSITE', 'o': 'eq'},
                                     {'a': '2', 'c': 'VSIM', 'o': 'eq'},
                                     {'a': '0', 'c': 'VSUB', 'o': 'eq'},
                                     {'a': '2009/09/03 17:53', 'c': 'VSTART', 'o': 'lt'},
                                     {'a': '2009/09/03 17:53', 'c': 'VEND', 'o': 'gt'}],
                               'o': 'and'}],
                         'o': 'and'}
                    }
    
    ee = ReContext(dd)()
    assert ee == orig , "the recontexted %s mismatches orig %s " % ( repr(ee) , repr(orig))
    ## problems with unicode cf integers
        
    
   
           
def test_sortlink():
    """
        sort links on the table headings are not context aware ... 
        they are created via tw.rum/templates/datagrid.html ... link_for_sort_key(col.name)
        using query.as_dict
        
        
        one way to workaround the problem ... is to amend the request dict to put it 
        back into the context aware form  
        
    """
    from webob import Request, UnicodeMultiDict
    raw = Request.blank("/SimPmtSpecDbis?q.c-1.c-2.o=eq&q.c-1.c-3.o=lt&q.c-1.c-3.c=VSTART&q.o=and&q.c-1.c-2.c=VSUB&q.c-1.c-2.a=0&q.c-1.c-4.a=2009%2F09%2F03+16%3A05&q.c-1.c-4.c=VEND&q.c-1.c-3.a=2009%2F09%2F03+16%3A05&q.c-1.c-4.o=gt&q.c-1.c-0.o=eq&q.c-1.c-1.o=eq&q.c-1.c-1.a=2&q.c-1.c-1.c=VSIM&q.c-1.c-0.a=1&q.c-1.c-0.c=VSITE&q.c-1.o=and&q.c-0.c-1.a=510&q.c-0.o=or&q.c-0.c-0.a=500&q.c-0.c-0.c=ROW&q.c-0.c-1.c=ROW&sort-0.dir=asc&q.c-0.c-1.o=lte&sort-0.c=DARKRATE&q.c-0.c-0.o=gt&limit=30")
    req = UnicodeMultiDict( raw.GET )
    from formencode import variabledecode
    d =  variabledecode.variable_decode(req)
    
    assert d == {'limit': u'30',
     'q': {'c': [{'c': [{'a': u'500', 'c': u'ROW', 'o': u'gt'},
                        {'a': u'510', 'c': u'ROW', 'o': u'lte'}],
                  'o': u'or'},
                 {'c': [{'a': u'1', 'c': u'VSITE', 'o': u'eq'},
                        {'a': u'2', 'c': u'VSIM', 'o': u'eq'},
                        {'a': u'0', 'c': u'VSUB', 'o': u'eq'},
                        {'a': u'2009/09/03 16:05', 'c': u'VSTART', 'o': u'lt'},
                        {'a': u'2009/09/03 16:05', 'c': u'VEND', 'o': u'gt'}],
                  'o': u'and'}],
           'o': u'and'},
     'sort': [{'c': u'DARKRATE', 'dir': u'asc'}]}

    expr = req2expr(d)
    print "expr %s" % repr(expr)
    #n = ReContext(d)()


class ctx_(Expression):
    """
        not a first class citizen as Expression.from_dict will not recreate this ... yet  
    """
  
from rum.query import simplify 
@simplify.when((ctx_,))
def _simplify_ctx(obj):
    """
        extend the simplify generic function such that this is used 
        by the Query class without subclassing ... changing the result of
        Query.as_dict when the expression features ctx_
    """
    v = obj.col
    if ctx_complete(v):
        n2a = ctx['_name2attr']
        return simplify(
           and_([ 
              eq(n2a['Site'],v['Site']) ,
              eq(n2a['SimFlag'],v['SimFlag']) ,
              eq(n2a['DetectorId'],v['DetectorId']) ,    
              lt(n2a['TimeStart'],v['Timestamp']),
              gt(n2a['TimeEnd'],v['Timestamp']),
                ])) 
    return {}






def test_ctx_as_dict():
    expr = ctx_({'Site':1, 'SimFlag':2 , 'DetectorId':0 , 'Timestamp':"2009/09/03 16:05"})
    q = Query(expr)
    d = q.as_dict()
    assert d == {'q': {'a': None,
           'c': [{'a': 1, 'c': 'VSITE', 'o': 'eq'},
                 {'a': 2, 'c': 'VSIM', 'o': 'eq'},
                 {'a': 0, 'c': 'VSUB', 'o': 'eq'},
                 {'a': '2009/09/03 16:05', 'c': 'VSTART', 'o': 'lt'},
                 {'a': '2009/09/03 16:05', 'c': 'VEND', 'o': 'gt'}],
           'o': 'and'}}


if __name__=='__main__':
    test_ctxonly()
    test_ctx_and_xtr()
    test_sortlink()
    test_ctx_as_dict()   
    test_ctxquery_with_xtr()
     
    
    