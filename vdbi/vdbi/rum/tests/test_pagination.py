
from vdbi.rum.query import *
from vdbi.rum.urltest import URLtest

def test_limit_offset():
    g = "http://localhost:6060/SimPmtSpecDbis?q.present=Table&q.plt.param.limit=500&q.plt.param.offset=0&q.plt.param.width=600&q.plt.o=plt_&q.plt.param.height=400&limit=30&offset=3150"
    u = URLtest(g)
    od = u.od
    q = u.q

    assert od.get('limit',None)
    assert od.get('offset',None)
    assert q.offset == int(od['offset'])
    assert q.limit  == int(od['limit'])


if __name__=='__main__':
    test_limit_offset()