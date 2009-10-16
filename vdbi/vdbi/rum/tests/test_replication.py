
from vdbi.rum.query import *
from vdbi.rum.query import _vdbi_expression, _vdbi_widget
from vdbi.rum.urltest import URLtest
from copy import copy


gurl = "http://localhost:6060/SimPmtSpecVlds?q.ctx.a=and&q.ctx.o=ctx_&q.ctx.c-0.SimFlag=2&q.ctx.c-0.Site=1&q.ctx.c-0.DetectorId=0&q.ctx.c-0.Timestamp=2009-10-09+18%3A22%3A45&q.xtr.a=xtr_&q.xtr.o=and&q.xtr.c-0.c=VSTART&q.xtr.c-0.o=eq&q.xtr.c-0.a=&q.present=Plot&q.present=Table&q.plt.o=plt_&q.plt.c-0.y=VSTART&q.plt.c-0.x=SEQ"
u = URLtest(gurl)



def test_replication():
    ## passage thru the eye of the flat_dict is needed for .json links to faithfully propagate the query 
    u = URLtest(gurl)
    r = u.repl
    assert r.od == u.od 
    ##assert r.raw.url == u.raw.url   ... fails due to variation in ordering of params 

def test_qclone():
    q = u.q
    kw= { 'limit':10,'offset':20,'sort':"womble"}
    c = q.clone(**kw)
    a = q.as_flat_dict_for_widgets()
    b = c.as_flat_dict_for_widgets()
    a.update(**kw) 
    assert a == b



if __name__=='__main__':
    test_replication()
    test_qclone()