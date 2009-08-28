
from rum import query as rumquery

"""
   http://pypi.python.org/pypi/prioritized_methods

"""


print "attempy generic funciton overide "


get = rumquery.QueryFactory.get.im_func

class DbiQueryFactory(rumquery.QueryFactory):
    
    
    #@get.when((object,), prio=1000 )
    @get.when()
    def _get(next_method, self, resource):
        print "next-method %s " % next_method
        return "hello"


if __name__=='__main__':
    qf = rumquery.QueryFactory()
    q = qf.get(object())
    print qf, q 