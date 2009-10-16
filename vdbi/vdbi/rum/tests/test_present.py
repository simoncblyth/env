from vdbi.rum.query import *
from vdbi.rum.query import _vdbi_expression, _vdbi_widget
from vdbi.rum.urltest import URLtest
from copy import copy


apres = [
   [],
   ['Summary'],
   ['Table'],
   ['Plot'],
   ['Summary','Table'],
   ['Summary','Plot'],
   ['Plot','Table'],
   ['Summary','Table','Plot'],
   ]
   
   
def test_present():
    for i,pres in enumerate(apres):
        s = "&".join(["q.present=%s" % p for p in pres])
        g = "http://localhost:6060/SimPmtSpecDbis?q.ctx.a=and&q.ctx.o=ctx_&q.xtr.a=xtr_&q.xtr.o=and&%s&q.plt.param.limit=500&q.plt.param.offset=0&q.plt.param.width=600&q.plt.param.height=400&q.plt.o=plt_" % s
        print i, pres
        u = URLtest(g)
        od = u.od
        q = u.q
        for p in pres:assert p in od['q']['present'] 
        qpl = q.present_list()
        assert sorted(qpl) == sorted(pres) 


if __name__=='__main__':
    test_present()


 
 



    