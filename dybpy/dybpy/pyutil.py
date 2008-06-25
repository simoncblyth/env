
import pprint


class PrintLogger:
    def hdr(self):
        return "<%s [%s] > " % ( self.__class__.__name__ , id(self) )
    def log(self, *args , **kwargs ):
        print "%s %s %s" % ( self.hdr() , " ".join(args),  pprint.pformat(kwargs) )

