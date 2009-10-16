import logging
log = logging.getLogger(__name__)
from copy import copy

from rum import app
from rum.query import *
from rum.query import _sort

from rum.query import _maybe_unicode

from formencode import variabledecode
from formencode.validators import Int, Invalid

from vdbi.dbg import debug_here
from vdbi.dyb import ctx
from vdbi.rum.param import present as present_
from vdbi import dbi_default_plot

class ctx_(Expression):
    """
        Expression.from_dict classmethod is overridden below  to conjure this 
        from the dict 
          
        The name setting is used via the metaclass to enable 
        cls.by_name used by Expression.from_dict 
    """
    name = "ctx_"
  
class plt_(Expression):
    name = "plt_"
    
  
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


@translate.when((plt_,))
def _plt(expr, resource):
    return None

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


@simplify.when((plt_,))
def _simplify_plt(obj):
    """
        extend the simplify generic function such that this is used 
        by the Query class without subclassing ... changing the result of
        Query.as_dict when the expression features plt_
    """   
    if isinstance(obj.col, (dict,)):
        c = [obj.col]
    elif isinstance(obj.col,(list,tuple,)):
        c = obj.col
    else:
        c = None

    if not(obj.arg):obj.arg = u"and"
    return { 'c':c , 'o':u"plt_" , 'a':obj.arg } 


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
    if cls.__name__ in ('ctx_','plt_'):return cls(*args)    ## special case to avoid recursing into the ctx_ or plt_
    return cls(*map(cls.from_dict, args))

Expression.from_dict = classmethod(_vdbi_expression_from_dict)


## these need the widget dict layout ... 

def _get_present(d):
    """
        CheckBoxList yields either a value or a list ... 
        regularize that to a list
    """
    if not('q' in d):return None
    present = d['q'].get('present',[]) 
    if type(present) != list:present=[present]
    return present

def _unget_present(p):
    """ 
         Feed the CheckBoxList back what it likes
    """
    if len(p) == 1:return p[0]
    elif len(p) > 1:return p
    elif len(p) == 0:return None    

_get_plotparam = lambda d:d.get('q',{}).get('plt',{}).get('param',{})

def _default_plt(d):
    if not('plt' in d['q']):
        d['q']['plt'] = {}
        
    ## make the plt sufficienctly well formed to yield the plt_ expression 
    ## that acts as a trojan horse to hold the parameters inside the query     
    if not('a' in d['q']['plt']):d['q']['plt']['a'] = {} 
    if not('c' in d['q']['plt']):d['q']['plt']['c'] = [] 
    if not('o' in d['q']['plt']):d['q']['plt']['o'] = u"plt_"


def _vdbi_comps(d):
    c = []
    if not 'q' in d:return c
    ## tuck the present away inside the 'plt' as [] , ['Table'] or ['Table','Plot']
    _default_plt(d)
          
    ## CONFUSING ..WHAT IS PLT DEFAULTING DOING HERE ?        
    d['q']['plt']['a'].update({ 'present' : _get_present(d) , 'param':_get_plotparam(d) })
    
    ## ctx and xtr are expression based ... they need to have a "c" if they are non-empty
    for br in ('ctx','xtr'):
        if br in d['q'] and 'c' in d['q'][br]:c.append( d['q'][br] )

    ## plt with content can have no "c"    
    for br in ('plt',):
        if br in d['q'] and 'a' in d['q'][br]:c.append( d['q'][br] )
    return c


def _vdbi_expression(d):
    """  
         convert widget dict into expression dict 
         (corresponding to the Rum original widget layout)
         
         cannot fake thinks to retain a consistent structure as the 
         fakes will appear in the interface ... so have to detect the 
         different dict topologies 
    """
    c = _vdbi_comps(d)
    if len(c) == 1:
        return { 'q':c[0] }
    elif len(c) > 1:
        return { 'q':{ 'o':u'and' , 'c':c }}    
    else:
        return d
    
    
def _vdbi_widget(od):
    """     
         convert expression dict (eg that returned from q.as_dict()) 
         into widget dict putting back the ctx/xtr/plt nodes
         
    """
    
    d = copy(od)
    
    if d and 'q' in d:    ## identify single branch queries : only xtr/ctx/plt 
        if   d['q'].get('a', None) == "xtr_":lab = "xtr"
        elif d['q'].get('o', None) == "ctx_":lab = "ctx"
        elif d['q'].get('o', None) == "plt_":lab = "plt"
        else:
            lab = None
                    
        brs = {}    
        if lab:
            brs[lab] = d['q']
        else:
            if len(d['q']['c']) > 0: 
                for el in d['q']['c']:
                    if el.get('o',None) and el['o'] in (u"ctx_",u"plt_"):
                        brs[str(el['o'])[0:-1]] = el  
                    else:
                        brs['xtr'] = el           
            else:
                print "_vdbi_uncast unsupported dict layout %s  " % (repr(d))
          
        ## untuck the present from its hiding place inside the plt
        if 'plt' in brs and 'a' in brs['plt']: 
            if 'present' in brs['plt']['a']:
                present = _unget_present( brs['plt']['a']['present'] )
                if present:
                    brs.update( {'present':present })
                del brs['plt']['a']['present']
            if 'param' in brs['plt']['a']:
                param = brs['plt']['a'].get('param', None)
                if param:
                    brs['plt']['param'] = param
                    del brs['plt']['a']['param']
        d['q'] = brs
    return d


Query.as_dict_for_widgets = lambda q:_vdbi_widget(q.as_dict())
     
def as_flat_dict_for_widgets(self):
    return variabledecode.variable_encode(self.as_dict_for_widgets(), add_repetitions=False)
Query.as_flat_dict_for_widgets = as_flat_dict_for_widgets

## OVERRIDE THE ORIGINAL THAT WAS YEILDING EXPRESSION DICT URLS
##  Query.as_flat_dict = as_flat_dict_for_widgets
      
            
def _vdbi_query_from_dict(cls, od):
    """Builds up a :class:`Query` object from a dictionary"""
    expr = sort = limit = offset = None
    od = variabledecode.variable_decode(od)
    d = _vdbi_expression(od)  
    #debug_here()
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
           
    obj = cls(expr, sort, limit, offset) 
    return obj

Query.from_dict = classmethod(_vdbi_query_from_dict)


def show_smth(self, smth="Plot" ):
    default = present_.get_default(smth)
    present = self.present_list()
    if not(present):return default
    if len(present) == 0:return default
    return smth in present
    
def present_list(self):
    dfw = self.as_dict_for_widgets()
    return _get_present( dfw )
    
def plotparam(self):
    dfw = self.as_dict_for_widgets()
    return _get_plotparam( dfw )
    
def plotseries(self):
    """"
     if no plot series is specified give the default       
    """
    dfw = self.as_dict_for_widgets()
    sdc = dfw.get('q',{}).get('plt',{}).get('c', [])
    if len(sdc) == 0:
        routes=app.request.routes
        sdc = dbi_default_plot( routes['resource'] )
    return sdc
    
Query.show_smth = show_smth
Query.show_plot = lambda self:self.show_smth("Plot")
Query.show_table = lambda self:self.show_smth("Table")
Query.show_summary = lambda self:self.show_smth("Summary")
Query.present_list = present_list
Query.plotparam = plotparam
Query.plotseries = plotseries


class DbiQueryFactory(QueryFactory): 
    def __call__(self, resource, request_args=None, **kw):
         log.debug("DbiQueryFactory.__call__ req = %s " % repr(request_args) ) 
         query = super(DbiQueryFactory, self).__call__(resource, request_args=request_args , **kw ) 
         log.debug("DbiQueryFactory ... query %s " % query ) 
         return query 



if __name__=='__main__':
    print "testing moved elsewhere"


    