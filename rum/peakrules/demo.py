"""
    http://siddhi.blogspot.com/2009/05/pattern-matching-with-peak-rules.html
"""

from peak.rules import abstract, when
from prioritized_methods import prioritized_when
@abstract()
def handle(path, method):
    pass

@prioritized_when(handle, "path == '/'", prio=1)
def not_a_resource(path, method):
    print "not a resource"

@when(handle, "method == 'GET'")
def get_resource(path, method):
    print "getting", path

@when(handle, "method == 'PUT'")
def create_resource(path, method):
    print "creating", path

@when(handle, "method == 'POST'")
def update_resource(path, method):
    print "updating", path




if __name__=='__main__':

     for meth in "GET PUT POST OTHER".split():
         for path in "/ hello".split():
             print path, meth 
     	     handle( path ,  meth )

