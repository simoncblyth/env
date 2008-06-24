

import GaudiPython as gp 


class irange(object):
    """  cranking the iterator from py side   TODO: yield a count in a tuple with the item  """
    def __init__(self, begin, end):
        self.begin, self.end = begin, end
        self.count = 0
    def __iter__(self):
        it = self.begin
        while it != self.end:
            yield it.__deref__()
            it.__postinc__(1)
            self.count += 1

def print_(o):
    """
         special handling to call print methods like :
             void GenParticle::print(basic_ostream<char,char_traits<char> >& ostr = std::cout)
          as print is a reserved word in python and as the output goes to a stream 
    """
    if hasattr(o,"print"):
        ss = gp.gbl.stringstream()
        __print = getattr(o, "print")
        __print(ss)
        return ss.str()
    return None


def fillStream_(o):    
    if hasattr(o,"fillStream"):
        ss = gp.gbl.stringstream()
        __fillStream = getattr(o, "fillStream")
        __fillStream(ss)
        return ss.str()
    return None


def format_(o):
    import pprint
    return pprint.pformat(o)
