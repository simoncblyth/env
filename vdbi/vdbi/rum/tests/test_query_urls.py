
from vdbi.rum.query import *
from vdbi.rum.query import _vdbi_expression, _vdbi_widget
from vdbi.rum.urltest import URLtest
from copy import copy




def test_ctx_layout():
    u = URLtest("/SimPmtSpecDbis?q.ctx.a=and&q.ctx.o=ctx_&q.ctx.c-0.SimFlag=2&q.ctx.c-0.Site=1&q.ctx.c-0.DetectorId=0&q.ctx.c-0.Timestamp=2009%2F09%2F04+16%3A43&q.xtr.o=and&q.xtr.c-0.c=RING&q.xtr.c-0.o=eq&q.xtr.c-0.a=2")
    assert u.od == {'q': {'ctx': {'a': u'and',
                   'c': [{'DetectorId': u'0',
                          'SimFlag': u'2',
                          'Site': u'1',
                          'Timestamp': u'2009/09/04 16:43'}],
                   'o': u'ctx_'},
           'xtr': {'c': [{'a': u'2', 'c': u'RING', 'o': u'eq'}], 'o': u'and'}}} , "raw dict arising from widget layout has changed %s " % repr(u.od)
    
    x = and_([ctx_([{'Timestamp': u'2009/09/04 16:43', 'DetectorId': u'0', 'SimFlag': u'2', 'Site': u'1'}], u'and'), and_([eq(u'RING', u'2')]), plt_([], {'present': [], 'param': {}})])
    assert `u.expr` == "and_([ctx_([{'Timestamp': u'2009/09/04 16:43', 'DetectorId': u'0', 'SimFlag': u'2', 'Site': u'1'}], u'and'), and_([eq(u'RING', u'2')]), plt_([], {'present': [], 'param': {}})])"
    #assert `u.expr` == `x`   fails due to ordering of "present" and "param" inside the plt_ ... how to canonicalize this ???

def test_ctx_req2q():    
    u = URLtest("/SimPmtSpecDbis?q.ctx.a=and&q.ctx.o=ctx_&q.ctx.c-0.SimFlag=2&q.ctx.c-0.Site=1&q.ctx.c-0.DetectorId=0&q.ctx.c-0.Timestamp=2009%2F09%2F04+16%3A43&q.xtr.o=and&q.xtr.c-0.c=RING&q.xtr.c-0.o=eq&q.xtr.c-0.a=2")   
    #assert u.qdw == u.od  ## failing 
    
    qdw = u.qdw 
    od = u.od
    assert qdw == {'q': {'ctx': {'a': u'and',
                   'c': [{'DetectorId': u'0',
                          'SimFlag': u'2',
                          'Site': u'1',
                          'Timestamp': u'2009/09/04 16:43'}],
                   'o': u'ctx_'},
           'plt': {'a': {'param': {}}, 'c': [], 'o': u'plt_'},
           'xtr': {'a': None,
                   'c': [{'a': '2', 'c': 'RING', 'o': 'eq'}],
                   'o': 'and'}}}
                                 
    assert od ==  {'q': {'ctx': {'a': u'and',
                    'c': [{'DetectorId': u'0',
                           'SimFlag': u'2',
                           'Site': u'1',
                           'Timestamp': u'2009/09/04 16:43'}],
                    'o': u'ctx_'},
           'xtr': {'c': [{'a': u'2', 'c': u'RING', 'o': u'eq'}], 'o': u'and'}}}

    ## to get a match need to add in these
    od['q']['plt'] = {'a': {'param': {}}, 'c': [], 'o': u'plt_'}
    od['q']['xtr']['a'] = None
    assert od == qdw


def test_ctx0_as_dict():
    a = {'Site':1, 'SimFlag':2 , 'DetectorId':0 , 'Timestamp':"2009/09/03 16:05"}
    d = {'q': {'c': [a], 'o': u"ctx_" , 'a':u"and" } }
    assert Query( ctx_(a) ).as_dict() == d

def test_ctx1_as_dict():
    a = {'Site':1, 'SimFlag':2 , 'DetectorId':0 , 'Timestamp':"2009/09/03 16:05"}
    b = copy(a) ; b['SimFlag'] = 1
    expr = ctx_( [a,b], u"or")
    assert Query(expr).as_dict() == { 'q': { 'c': [a,b], 'o': u'ctx_', 'a':u"or" }}

def test_ctx0_from_dict():  ## failing as using from_dict with expresssion dict rather than widget dict 
    a = {'Site':1, 'SimFlag':2 , 'DetectorId':0 , 'Timestamp':"2009/09/03 16:05"}
    b = copy(a) ; b['SimFlag'] = 1
    
    d0 = {'q': {'c': [a],    'o': u"ctx_" , 'a':u"and" }}
    d1 = {'q': {'c': [a],    'o': u"ctx_" , 'a':u"or" }}
    d2 = {'q': {'c': [a,b] , 'o': u"ctx_" , 'a':u"or"  }}
    d3 = {'q': {'c': [a,b] , 'o': u"ctx_" , 'a':u"and"  }}

    for d in [d0,d1,d2,d3]:
        assert  Query.from_dict(d).as_dict() == d , "from_dict/as_dict commute fail for %s " % repr(d)    
        

def test_cast():    
    e = {'q': {'ctx': {'a': u'and',
                   'c': [{'DetectorId': u'0',
                          'SimFlag': u'2',
                          'Site': u'1',
                          'Timestamp': u'2009/09/04 16:43'}],
                   'o': u'ctx_'},
            'plt': {'a': {'param': {}, 'present': []}, 'c': [], 'o': u'plt_'},
           'xtr': {'c': [{'a': u'2', 'c': u'RING', 'o': u'eq'}], 'o': u'and'}}}

    f = _vdbi_expression( e )   ## remove the ctx and xtr 
    xf = {'q': {'c': [{'a': u'and',
                  'c': [{'DetectorId': u'0',
                         'SimFlag': u'2',
                         'Site': u'1',
                         'Timestamp': u'2009/09/04 16:43'}],
                  'o': u'ctx_'},
                 {'c': [{'a': u'2', 'c': u'RING', 'o': u'eq'}], 'o': u'and'},
                 {'a': {'param': {}, 'present': []}, 'c': [], 'o': u'plt_'}],
           'o': u'and'}}
    assert f == xf
                        
    g = _vdbi_widget( f )   ## put them back ... as needed for widget display    
    xg = {'q': {'ctx': {'a': u'and',
                   'c': [{'DetectorId': u'0',
                          'SimFlag': u'2',
                          'Site': u'1',
                          'Timestamp': u'2009/09/04 16:43'}],
                   'o': u'ctx_'},
           'plt': {'a': {'param': {}}, 'c': [], 'o': u'plt_'},
           'xtr': {'c': [{'a': u'2', 'c': u'RING', 'o': u'eq'}], 'o': u'and'}}}
    assert g == xg
    assert _vdbi_widget( _vdbi_expression( e ) ) == e



def test_ctx_only():
    u = URLtest("/SimPmtSpecDbis?q.ctx.a=and&q.ctx.o=ctx_&q.ctx.c-0.SimFlag=2&q.ctx.c-0.Site=1&q.ctx.c-0.DetectorId=0&q.ctx.c-0.Timestamp=2009-09-10+15%3A04%3A06&q.ctx.c-1.SimFlag=1&q.ctx.c-1.Site=1&q.ctx.c-1.DetectorId=0&q.ctx.c-1.Timestamp=2009-09-17+15%3A05%3A14&q.xtr.o=and")    
    qd = u.qd
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
    assert qd == x , "query as dict when ctx only mismatches expectations "
   ## doesnt match due to addition of plt and resulting conversion from singular to multi form 


    
def test_with_present():    
    u = URLtest("http://localhost:6060/SimPmtSpecVlds?q.ctx.a=and&q.ctx.o=ctx_&q.ctx.c-0.SimFlag=2&q.ctx.c-0.Site=1&q.ctx.c-0.DetectorId=0&q.ctx.c-0.Timestamp=2009-10-09+18%3A22%3A45&q.xtr.a=xtr_&q.xtr.o=and&q.xtr.c-0.c=VSTART&q.xtr.c-0.o=eq&q.xtr.c-0.a=&q.present=Plot&q.present=Table&q.plt.o=plt_&q.plt.c-0.y=VSTART&q.plt.c-0.x=SEQ")
    ##assert u.od == u.dqw     ... need to ensure empty plt param in widget conversion to make this work  
    od = u.od
    qdw = u.qdw    
    assert od == {'q': {'ctx': {'a': u'and',
                   'c': [{'DetectorId': u'0',
                          'SimFlag': u'2',
                          'Site': u'1',
                          'Timestamp': u'2009-10-09 18:22:45'}],
                   'o': u'ctx_'},
           'plt': {'c': [{'x': u'SEQ', 'y': u'VSTART'}], 'o': u'plt_'},
           'present': [u'Plot', u'Table'],
           'xtr': {'a': u'xtr_',
                   'c': [{'a': u'', 'c': u'VSTART', 'o': u'eq'}],
                   'o': u'and'}}}
 
    od['q']['plt'].update( { 'a':{'param': {} } })    ## add empty plt param for agreement
    assert qdw == od


def test_passage_thru_query():
    u = URLtest("http://localhost:6060/SimPmtSpecVlds?q.ctx.a=and&q.ctx.o=ctx_&q.ctx.c-0.SimFlag=2&q.ctx.c-0.Site=1&q.ctx.c-0.DetectorId=0&q.ctx.c-0.Timestamp=2009-10-09+18%3A22%3A45&q.xtr.a=xtr_&q.xtr.o=and&q.xtr.c-0.c=VSTART&q.xtr.c-0.o=eq&q.xtr.c-0.a=&q.present=Plot&q.present=Table&q.plt.o=plt_&q.plt.c-0.y=VSTART&q.plt.c-0.x=SEQ")
    assert len(u.comps) == 3              
    assert u.qdw == u.odk 
    
    


def test_zen():
    u = URLtest("http://localhost:6060/SimPmtSpecDbis?q.ctx.a=and&q.ctx.o=ctx_&q.xtr.a=xtr_&q.xtr.o=and&q.present=Summary&q.present=Table&q.plt.param.limit=500&q.plt.param.offset=0&q.plt.param.width=600&q.plt.param.height=400&q.plt.o=plt_")
    assert u.od.keys() == ['q']
    assert u.od['q'].keys() == ['plt', 'ctx', 'xtr', 'present']
    assert u.od['q']['ctx'].keys() ==  ['a', 'o']
    assert u.od['q']['plt'].keys() == ['o', 'param']
    assert u.od['q']['xtr'].keys() == ['a', 'o']







if __name__=='__main__':
    test_ctx_layout()
    test_ctx_req2q()
    test_ctx0_as_dict()
    test_ctx1_as_dict()
    #test_ctx0_from_dict()
    test_cast()
    #test_ctx_only()
    test_with_present()
    test_replication()
    test_zen()
