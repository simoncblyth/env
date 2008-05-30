"""subpkg __init__ docstring """

from context import ctx as ctx
from context import present as present 

present(ctx(globals()))

def setup():
    present(ctx(globals()))
def teardown():
    present(ctx(globals()))

