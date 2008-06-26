
import pprint


def reload_():
    import sys
    reload(sys.modules[__name__])

class PrintLogger:
    def hdr(self):
        return "<%s [0x%08X] > " % ( self.__class__.__name__ , id(self) )
    def log(self, *args , **kwargs ):
        print "%s %s %s" % ( self.hdr() , " ".join(args),  pprint.pformat(kwargs) )

