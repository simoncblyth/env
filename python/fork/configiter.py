"""
simon:fork blyth$ nosetests configiter.py  -v
configiter.test_config('base', {'red': 'red'}) ... ok
configiter.test_config('cust', {'green': 'green', 'red': 'red'}) ... ok
configiter.test_config('cust2', {'blue': 'blue', 'green': 'green', 'red': 'red'}) ... ok

----------------------------------------------------------------------
Ran 3 tests in 0.008s

OK
                 nosetests configiter.py  -v -s       TO SEE STDOUT
"""

import pprint
g = None


def payload(*args,**kwargs):
    print "payload\n" + "\n".join([repr(a) for a in args]) + pprint.pformat(kwargs)
    print "global\n" + pprint.pformat(g)
    return 0 

def simple_runner(*args,**kwargs):
    assert callable(args[0])
    return args[0](args[1:])

def _cfg(*yrgs):
    """ helper func for consistent yielding a structure acceptable for nosetest running """ 
    r = [simple_runner,payload]
    r.extend(yrgs)
    return r

def configiter():
    """ this is a generator function returning multiple variations 
        of configuration 
        when iterated over  ... the tuple yielded  provides a callablep and its arguments
        note the actual config g is global ... so it does not need to be returned
    """
    global g
    g = {}
    g['red'] = "red"
    yield  _cfg("base" , g)  
    g['green'] = "green"
    yield  _cfg("cust",  g) 
    g['blue' ] = "blue"
    yield  _cfg("cust2",  g)  
    
def test_config():
    ## the callables yielded are run by the nose testrunner
    for cfg in configiter():
        print "test_config %s " % repr(cfg)
        yield cfg

def confloop(cit):
    """ not used by the testing """
    global g 
    for cfg in cit:
        print "cfg... %s " % repr(cfg)
        assert callable(cfg[0])  , "1st in tuple must be callable  " 
        assert callable(cfg[1])  , "2nd in tuple must be callable  "
        assert cfg[3] is g , " %s is not g " % repr(cfg[3])
        cfg[0](cfg[1],cfg[2:])

if __name__=='__main__':
    confloop(configiter())
        