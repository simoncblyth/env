
from tech import Tech
from qry import Qry


def check_technique(callable, args, kwargs):
    callable(*args, **kwargs)

def test_techniques():
    q = Qry( "DcsPmtHv" , read_default_group="client" , limit=60000, verbose=1, method=1 )

    args = [ q ]
    kwargs = dict()
    for t in Tech.callables():
        yield  check_technique, t,  args, kwargs


if __name__ == '__main__':
    pass
    print Tech.subclasses()
    print Tech.callables()
    print Tech.names()

    for chk, callable, args, kwargs in test_techniques():
        chk( callable, args, kwargs)
 
