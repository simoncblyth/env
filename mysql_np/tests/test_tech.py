
from tech import Tech
from env.dyb.db import Qry


def check_technique(callable, args, kwargs):
    a = callable(*args, **kwargs)
    print repr(a)

def test_techniques():
    q = Qry( "DcsPmtHv" , read_default_group="client" , limit=60000, verbose=1, method=1 )

    args = [ q ]
    kwargs = dict(verbose=1)
    for t in Tech.callables():
        yield  check_technique, t,  args, kwargs


if __name__ == '__main__':
    pass
    print Tech.callables()
    print Tech.names()

    for chk, callable, args, kwargs in test_techniques():
        chk( callable, args, kwargs)
        
 
