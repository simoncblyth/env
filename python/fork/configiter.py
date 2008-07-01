"""
    DEVELOPMENT ABANDONED DUE TO DISCOVERY OF "InsulateRunner" NOSE PLUGIN

   This was a stab at a forking test runner 
      ... it works when all tests succeed !
      
   Unfortunately when there are failures, it is severely buggered with 
   recursive forking and overreporting ... the reports being 
   triggered by the exits of the children
   Plus ..   the inner nose reports status correctly ???
   the outer one sees everything OK

       # config     # runs        # reports 
          1           1 r              2
          2           3 r/rg/rg        4  
          3           7                8
              r/rg/rgb/rgb/rg/rgb/rgb
  
    Looking for a plugin that has solved these problems already, find...
  
  **nosepipe** gives "--with-process-isolation" , but it gets confused by generative tests

         nosetests configiter.py -v --with-process-isolation -s --with-xml-output --xml-outfile out.xml
        
  **InsulateRunner**  gives "--with-insulate" ,  yep it looks good... using master/slave pattern over socket
                                
         nosetests configiter.py -v -s --with-insulate --with-xml-output --xml-outfile=out.xml
 
"""

import pprint
import os
import time

g = None

#import runner

def payload(*args,**kwargs):
    print "payload\n" + "\n".join([repr(a) for a in args]) + pprint.pformat(kwargs)
    print "global\n" + pprint.pformat(g)
    if "blue" in g:
        import gibberish
    print "payload sleeping " 
    time.sleep(1)
    return 0 

#def run(*args,**kwargs):
#    runner.forking_runner( payload, *args, **kwargs)


def configiter():
    """ this is a generator function returning multiple variations 
        of configuration 
        when iterated over  ... the tuple yielded  provides a callablep and its arguments
        note the actual config g is global ... so it does not need to be returned
    """
    global g
    g = {}
    g['red'] = "red"
    yield  (payload,"base" , g)  
    g['green'] = "green"
    yield  (payload,"cust",  g) 
    g['blue' ] = "blue"
    yield  (payload,"cust2",  g)  
    

def test_conf():
    for cnf in configiter():
        yield cnf


def confloop(cit):
    """ not used by the testing """
    global g 
    for cfg in cit:
        print "cfg... %s " % repr(cfg)
        assert callable(cfg[0])  , "1st in tuple must be callable  " 
        assert cfg[2] is g , " %s is not g " % repr(cfg[2])
        cfg[0](cfg[1:])

if __name__=='__main__':
    confloop(test_configiter())
        