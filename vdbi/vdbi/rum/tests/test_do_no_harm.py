from vdbi.rum.query import *
from vdbi.rum.query import _vdbi_expression, _vdbi_widget
from vdbi.rum.urltest import URLtest
from copy import copy


g = "http://localhost:6060/SimPmtSpecVlds?q.ctx.a=and&q.ctx.o=ctx_&q.ctx.c-0.SimFlag=2&q.ctx.c-0.Site=1&q.ctx.c-0.DetectorId=0&q.ctx.c-0.Timestamp=2009-10-09+18%3A22%3A45&q.xtr.a=xtr_&q.xtr.o=and&q.xtr.c-0.c=VSTART&q.xtr.c-0.o=eq&q.xtr.c-0.a=&q.present=Plot&q.present=Table&q.plt.o=plt_&q.plt.c-0.y=VSTART&q.plt.c-0.x=SEQ"
u = URLtest(g)
q = u.q
oqd = q.as_dict()
dfw = q.as_dict_for_widgets()
nqd = q.as_dict()
assert oqd == nqd





# In [33]: q.as_dict()
# Out[33]: 
# {'q': {'a': {'param': {'height': u'400',
#                        'limit': u'500',
#                        'offset': u'0',
#                        'width': u'600'},
#              'present': [u'Summary', u'Table']},
#        'c': [],
#        'o': u'plt_'}}
# 
# In [34]: q
# Out[34]: Query(plt_([], {'present': [u'Summary', u'Table'], 'param': {'width': u'600', 'height': u'400', 'limit': u'500', 'offset': u'0'}}), None, None, None)
# 
# In [35]: q.as_dict_for_widgets()
# Out[35]: 
# {'q': {'plt': {'a': {},
#                'c': [],
#                'o': u'plt_',
#                'param': {'height': u'400',
#                          'limit': u'500',
#                          'offset': u'0',
#                          'width': u'600'}},
#        'present': [u'Summary', u'Table']}}
# 
# In [36]: q
# Out[36]: Query(plt_([]), None, None, None)
# 
# In [37]: q.as_dict_for_widgets??
# Type:           instancemethod
# Base Class:     <type 'instancemethod'>
# String Form:    <bound method Query.<lambda> of Query(plt_([]), None, None, None)>
# Namespace:      Interactive
# File:           /Users/blyth/env/vdbi/vdbi/rum/query.py
# Definition:     q.as_dict_for_widgets(q)
# Source:
# Query.as_dict_for_widgets      = lambda q:_vdbi_widget(q.as_dict())
# 
# In [38]: _vdbi_widget??
# 
# In [39]: q.as_dict()
# Out[39]: {'q': {'a': u'and', 'c': [], 'o': u'plt_'}}