"""
   http://pypi.python.org/pypi/prioritized_methods

   http://peak.telecommunity.com/DevCenter/RulesReadme
   http://peak.telecommunity.com/DevCenter/CombiningResults
   http://peak.telecommunity.com/DevCenter/RulesDesign

   http://www.mail-archive.com/turbogears-trunk@googlegroups.com/msg05137.html

"""

from rum import query as rumquery
#get = rumquery.QueryFactory.get.im_func

class DbiQueryFactory(rumquery.QueryFactory):
 
#    
#    @get.when("hasattr(resource,'yellow')")
#    def _get_query_cls_for_resource(self, resource):
#        print " %s " % resource
#        return "hello"

    def __call__(self, resource, request_args=None, **kw):
         query = super(DbiQueryFactory, self).__call__(resource, request_args=request_args, **kw )
         print "DbiQueryFactory __call__ override puts the query at my mercy  self:%s query:%s req:%s " % ( self , query , repr(request_args) )
         return query 


class A(object):pass
class B(object):
    yellow = True


if __name__=='__main__':
    
    
    a = A()
    b = B()
    
    dqf = DbiQueryFactory()
    qa = dqf.get(a)
    print "qa %s " % qa
    
    qb = dqf.get(b)
    print "qb %s " % qb

    