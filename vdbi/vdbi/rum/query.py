

from rum import query as rumquery
from formencode import variabledecode


class DbiQueryFactory(rumquery.QueryFactory):
 
    def __call__(self, resource, request_args=None, **kw):
         query = super(DbiQueryFactory, self).__call__(resource, request_args=request_args, **kw )
         d = variabledecode.variable_decode(request_args)
         print "DbiQueryFactory __call__ override puts the query at my mercy  self:%s query:%s req:%s d:%s" % ( self , query , repr(request_args) , repr(d))
         return query 



    
 
    