

from rum.query import *
from rum.query import _sort
from formencode import variabledecode
from formencode.validators import Int, Invalid

from vdbi import debug_here
from vdbi.dyb import ctx

import logging
log = logging.getLogger(__name__)


class ctx_(Expression):
    """
        Expression.from_dict classmethod is overridden below  to conjure this 
        from the dict 
          
        The name setting is used via the metaclass to enable 
        cls.by_name used by Expression.from_dict 
    """
    name = "ctx_"
  
  
from rumalchemy.query import translate   ## a little unhealthy importing from rumalchemy ?
@translate.when((ctx_,))
def _ctx(expr, resource):
    
    from vdbi.dyb import ctx
    from rum.query import and_, eq, lt, gt
    n2a = ctx['_name2attr']
    #debug_here()
    if not(expr.arg):expr.arg=u"and"
    
    xs = []
    for v in expr.col:
        xs.append( and_([ 
            eq(n2a['Site'],      Int(min=0).to_python(v['Site']) ) ,
            eq(n2a['SimFlag'],   Int(min=0).to_python(v['SimFlag']) ) ,    
            eq(n2a['DetectorId'],Int(min=0).to_python(v['DetectorId'])) ,    
            lt(n2a['TimeStart'], v['Timestamp']),
            gt(n2a['TimeEnd'],   v['Timestamp']),
         ]) ) 
             
    xpr = None
    if len(xs) == 1:
        xpr = xs[0] 
    elif len(xs) > 1:
        if unicode(expr.arg) == u"and":
            xpr = and_( xs )
        elif unicode(expr.arg) == u"or":
            xpr = or_( xs )  
        else:
            xpr = None
    return translate( xpr , resource )


  
from rum.query import simplify 
@simplify.when((ctx_,))
def _simplify_ctx(obj):
    """
        extend the simplify generic function such that this is used 
        by the Query class without subclassing ... changing the result of
        Query.as_dict when the expression features ctx_
    """   
    if isinstance(obj.col, (dict,)):
        c = [obj.col]
    elif isinstance(obj.col,(list,tuple,)):
        c = obj.col
    else:
        c = None
        
    if not(obj.arg):obj.arg = u"and"
    return { 'c':c , 'o':u"ctx_" , 'a':obj.arg } 


from rum.query import _maybe_unicode
def _vdbi_expression_from_dict(cls, d):
    """
       Override of the original Expression.from_dict 
       in order to stop the recursion when reach the ctx_ 
       expression, changing Query.from_dict
    """
    if isinstance(d, (list,tuple)):
        return map(cls.from_dict, d)
    if not isinstance(d, dict):
        # stops recursion at leaves
        return d
    cls = cls.by_name(d['o'])
    col = d['c']
    arg = d.get('a')
    args = [col, _maybe_unicode(arg)]
    if cls.__name__ == 'ctx_':return cls(*args)    ## special case to avoid recursing into the ctx_
    return cls(*map(cls.from_dict, args))

Expression.from_dict = classmethod(_vdbi_expression_from_dict)


def _vdbi_recast(d):
    """  
         recast the dict into the oldform without the ctx and xtr nodes 
         despite the new widget layout  
         
         cannot fake thinks to retain a consistent structure as the 
         fakes will appear in the interface ... so have to detect the 
         different dict topologies 
    """
    xtr = 'q' in d and 'xtr' in d['q'] and 'c' in d['q']['xtr']
    ctx = 'q' in d and 'ctx' in d['q'] and 'c' in d['q']['ctx']
    if xtr and ctx:
        return { 'q':{ 'o':u'and' , 'c':[ d['q']['ctx'] , d['q']['xtr'] ] }}
    elif ctx:
        return { 'q': d['q']['ctx']  }
    elif xtr:
        return { 'q': d['q']['xtr']  } 
    else:
        return d
    
def _vdbi_uncast(d):
    """ 
        Note :
           * planted a hidden field in "a" slot with value "xtr_"  for the extras
             to avoid divination 
       
         convert back to the form needed by the widgets 
         replacing the ctx and xtr nodes
         
    """
    if d and 'q' in d:    ## identify only xtr or only ctx queries
        if d['q'].get('a', None) and d['q']['a'] == "xtr_":
            lab = "xtr"
        elif d['q'].get('o', None) and d['q']['o'] == "ctx_":
            lab = "ctx"
        else:
            lab = None
            
        if lab:
            r =  { 'q':{ lab:d['q'] } }
            print "_vdbi_uncast %s ... return %s " % (repr(d), repr(r))
            return r
        else:
            if len(d['q']['c']) == 2 and d['q']['c'][0].get('o', None) and d['q']['c'][0]['o'] == u"ctx_":
                return { 'q':{ 'ctx': d['q']['c'][0] , 'xtr': d['q']['c'][1] }  }
            else:
                print "_vdbi_uncast unsupported dict layout %s  " % (repr(d))
    else:
        return d

            
def _vdbi_query_from_dict(cls, od):
    """Builds up a :class:`Query` object from a dictionary"""
    expr = sort = limit = offset = None
    od = variabledecode.variable_decode(od)
    d = _vdbi_recast(od)  
    if 'q' in d and 'c' in d['q']:
        expr = Expression.from_dict(d['q'])
    if 'sort' in d:
        sort = d['sort']
        if sort:
            sort = map(_sort.from_dict, sort)
    if 'limit' in d:
        limit = Int(min=0).to_python(d['limit'])
    if 'offset' in d:
        offset = Int(min=0).to_python(d['offset'])
    return cls(expr, sort, limit, offset)

Query.from_dict = classmethod(_vdbi_query_from_dict)


class DbiQueryFactory(QueryFactory): 
    def __call__(self, resource, request_args=None, **kw):
         log.debug("DbiQueryFactory.__call__ req = %s " % repr(request_args) ) 
         query = super(DbiQueryFactory, self).__call__(resource, request_args=request_args , **kw ) 
         log.debug("DbiQueryFactory ... query %s " % query ) 
         return query 





def test_new_layout_with_xtr_mark():
    """
       avoid divination of the dict layouts due to the presence of the xtr_ marker hailing from a hidden field 
    """
    from webob import Request, UnicodeMultiDict
    raw = Request.blank("/SimPmtSpecDbis?q.ctx.a=and&q.ctx.o=ctx_&q.ctx.c-0.SimFlag=2&q.ctx.c-0.Site=1&q.ctx.c-0.DetectorId=0&q.ctx.c-0.Timestamp=2009-09-10+20%3A10%3A01&q.xtr.a=xtr_&q.xtr.o=and&q.xtr.c-0.c=ROW&q.xtr.c-0.o=eq&q.xtr.c-0.a=10")
    req = UnicodeMultiDict( raw.GET )
    
    from formencode import variabledecode
    o = variabledecode.variable_decode(req)
    assert o == {'q': {'ctx': {'a': u'and',
                               'c': [{'DetectorId': u'0','SimFlag': u'2','Site': u'1','Timestamp': u'2009-09-10 20:10:01'}],
                               'o': u'ctx_'},
                       'xtr': {'a': u'xtr_',
                               'c': [{'a': u'10', 'c': u'ROW', 'o': u'eq'}],
                               'o': u'and'}}}
    
    q = Query.from_dict( req )
    d = q.as_dict()
    assert d == {'q': {'a': None,
                       'c': [
                             {'a': u'and',
                              'c': [{'DetectorId': u'0','SimFlag': u'2','Site': u'1','Timestamp': u'2009-09-10 20:10:01'}],
                              'o': u'ctx_'},
                             {'a': 'xtr_',
                              'c': [{'a': '10', 'c': 'ROW', 'o': 'eq'}],
                              'o': 'and'}
                            ],
                       'o': 'and'}}


def test_ctx_layout():
    from webob import Request, UnicodeMultiDict
    raw = Request.blank("/SimPmtSpecDbis?q.ctx.a=and&q.ctx.o=ctx_&q.ctx.c-0.SimFlag=2&q.ctx.c-0.Site=1&q.ctx.c-0.DetectorId=0&q.ctx.c-0.Timestamp=2009%2F09%2F04+16%3A43&q.xtr.o=and&q.xtr.c-0.c=RING&q.xtr.c-0.o=eq&q.xtr.c-0.a=2")
    req = UnicodeMultiDict( raw.GET )
    from formencode import variabledecode
    d =  variabledecode.variable_decode(req)
    
    e = {'q': {'ctx': {'a': u'and',
                   'c': [{'DetectorId': u'0',
                          'SimFlag': u'2',
                          'Site': u'1',
                          'Timestamp': u'2009/09/04 16:43'}],
                   'o': u'ctx_'},
           'xtr': {'c': [{'a': u'2', 'c': u'RING', 'o': u'eq'}], 'o': u'and'}}}
    
    assert d == e , "raw dict arising from widget layout has changed %s " % repr(d)
    n = { 'o':u'and' , 'c':[ d['q']['ctx'] , d['q']['xtr'] ] }   ## manual recast 
    x = and_([ctx_([{'Timestamp': u'2009/09/04 16:43', 'DetectorId': u'0', 'SimFlag': u'2', 'Site': u'1'}], u'and'), and_([eq(u'RING', u'2')])])
    expr = Expression.from_dict(n)
    assert expr ==  x , "changed expression %s " % repr()

def test_ctx_req2q():
    from webob import Request, UnicodeMultiDict
    raw = Request.blank("/SimPmtSpecDbis?q.ctx.a=and&q.ctx.o=ctx_&q.ctx.c-0.SimFlag=2&q.ctx.c-0.Site=1&q.ctx.c-0.DetectorId=0&q.ctx.c-0.Timestamp=2009%2F09%2F04+16%3A43&q.xtr.o=and&q.xtr.c-0.c=RING&q.xtr.c-0.o=eq&q.xtr.c-0.a=2")
    req = UnicodeMultiDict( raw.GET )
    q = Query.from_dict( req )         ## the recast is done in the overridden classmethod
    e = Query(and_([ctx_([{'Timestamp': u'2009/09/04 16:43', 'DetectorId': u'0', 'SimFlag': u'2', 'Site': u'1'}], u'and'), and_([eq(u'RING', u'2')])]), None, None, None)
    assert repr(q) == repr(e)
    d = q.as_dict()
    assert Query.from_dict(d).as_dict() == d

def test_ctx0_as_dict():
    cta = {'Site':1, 'SimFlag':2 , 'DetectorId':0 , 'Timestamp':"2009/09/03 16:05"}
    exp0 = ctx_( cta )
    exp1 = ctx_( cta , u"and")
    d = {'q': {'c': [cta], 'o': u"ctx_" , 'a':u"and" } }
    assert Query(exp0).as_dict() == d
    assert Query(exp1).as_dict() == d

def test_ctx1_as_dict():
    cta = {'Site':1, 'SimFlag':2 , 'DetectorId':0 , 'Timestamp':"2009/09/03 16:05"}
    from copy import copy
    ctb = copy(cta) ; ctb['SimFlag'] = 1
    expr = ctx_( [cta,ctb], u"or")
    assert Query(expr).as_dict() == { 'q': { 'c': [cta,ctb], 'o': u'ctx_', 'a':u"or" }}

def test_ctx0_from_dict():
    cta = {'Site':1, 'SimFlag':2 , 'DetectorId':0 , 'Timestamp':"2009/09/03 16:05"}
    from copy import copy
    ctb = copy(cta) ; ctb['SimFlag'] = 1
    
    d0 = {'q': {'c': [cta],      'o': u"ctx_" , 'a':u"and" }}
    d1 = {'q': {'c': [cta],      'o': u"ctx_" , 'a':u"or" }}
    d2 = {'q': {'c': [cta,ctb] , 'o': u"ctx_" , 'a':u"or"  }}
    d3 = {'q': {'c': [cta,ctb] , 'o': u"ctx_" , 'a':u"and"  }}

    for d in [d0,d1,d2,d3]:
        assert  Query.from_dict(d).as_dict() == d , "from_dict/as_dict commute fail for %s " % repr(d)
    

def test_cast():    
    e = {'q': {'ctx': {'a': u'and',
                   'c': [{'DetectorId': u'0',
                          'SimFlag': u'2',
                          'Site': u'1',
                          'Timestamp': u'2009/09/04 16:43'}],
                   'o': u'ctx_'},
           'xtr': {'c': [{'a': u'2', 'c': u'RING', 'o': u'eq'}], 'o': u'and'}}}

    f = _vdbi_recast( e )   ## remove the ctx and xtr 
    assert f == {'q': {'c': [{'a': u'and',
                   'c': [{'DetectorId': u'0',
                          'SimFlag': u'2',
                          'Site': u'1',
                          'Timestamp': u'2009/09/04 16:43'}],
                   'o': u'ctx_'},
                  {'c': [{'a': u'2', 'c': u'RING', 'o': u'eq'}], 'o': u'and'}],
            'o': u'and'}}
        
    g = _vdbi_uncast( f )   ## put them back ... as needed for widget display
    assert g ==         {'q': {'ctx': {'a': u'and',
                           'c': [{'DetectorId': u'0',
                                  'SimFlag': u'2',
                                  'Site': u'1',
                                  'Timestamp': u'2009/09/04 16:43'}],
                           'o': u'ctx_'},
                   'xtr': {'c': [{'a': u'2', 'c': u'RING', 'o': u'eq'}], 'o': u'and'}}}
    assert _vdbi_uncast( _vdbi_recast( e ) ) == e


def test_ctx_only():
    """
         when no additional criteria are added ... leaving just a ctx query 
         the form of the dict is causing assertions / key errors in the uncast 
    
    """
    from webob import Request, UnicodeMultiDict
    raw = Request.blank("/SimPmtSpecDbis?q.ctx.a=and&q.ctx.o=ctx_&q.ctx.c-0.SimFlag=2&q.ctx.c-0.Site=1&q.ctx.c-0.DetectorId=0&q.ctx.c-0.Timestamp=2009-09-10+15%3A04%3A06&q.ctx.c-1.SimFlag=1&q.ctx.c-1.Site=1&q.ctx.c-1.DetectorId=0&q.ctx.c-1.Timestamp=2009-09-17+15%3A05%3A14&q.xtr.o=and")    
    req = UnicodeMultiDict( raw.GET )
    q = Query.from_dict( req )
    print q
    d = q.as_dict()
    x = {'q': {'a': u'and',
               'c': [{'DetectorId': u'0',
                      'SimFlag': u'2',
                      'Site': u'1',
                      'Timestamp': u'2009-09-10 15:04:06'},
                     {'DetectorId': u'0',
                      'SimFlag': u'1',
                      'Site': u'1',
                      'Timestamp': u'2009-09-17 15:05:14'}],
                'o': u'ctx_'}}
    assert d == x , "query as dict when ctx only mismatches expectations "
    
    y = _vdbi_uncast( x )







if __name__=='__main__':

    test_ctx0_as_dict()  
    test_ctx1_as_dict()  
    test_ctx0_from_dict()
    
    test_ctx_layout()
    test_ctx_req2q()
    test_cast()
    
    test_ctx_only()
    test_new_layout_with_xtr_mark()
    