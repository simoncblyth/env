#!/usr/bin/env python
import os, logging, json
log = logging.getLogger(__name__)


class DAEChromaMap(object):

    @classmethod 
    def fromjson(cls, config):
        cmm = cls(config)  
        cmm.code2name = cmm.read()
        return cmm

    @classmethod
    def compare(cls, a, b ):
        a = a.code2name
        b = b.code2name
        assert len(a) == len(b) , "key mismatch"
        mismatch = 0 
        for k in a:
            same = a[k] == b[k]
            mkr = "" if same else "********"
            if not same:mismatch += 1
            print "%2d %-20s %-20s %s" % ( k, a[k], b[k], mkr )
        pass
        log.info("compare sees %s mismatches " % mismatch )
        return mismatch

    def dump(self):
        log.info("dump")
        print self

    def __init__(self, config, c2n=None):
        self.config = config
        self.code2name = c2n

    def __str__(self):
        d = self.code2name 
        return "\n".join(["%3d %s " % (k, d[k]) for k in sorted(d.keys(),key=lambda _:_)])

    def write(self):
        if os.path.exists(self.path):
            prior = self.fromjson(self.config) 
            mismatch = self.compare( prior, self )
        pass
        log.info("writing to %s " % self.path )
        with open(self.path,"w") as fp:
            json.dump(self.code2name, fp) 

    def read(self):
        """ 
        json keys and values are unicode strings by default, 
        so convert to int,str on reading to match the creation dict 
        """
        if not os.path.exists(self.path):
            log.warn("no such path %s " % self.path)
            return
        log.info("reading from %s " % self.path )
        with open(self.path,"r") as fp:
            pd = json.load(fp)
        pass
        return dict(map(lambda _:(int(_[0]),str(_[1])),pd.items()))


if __name__ == '__main__':
    pass



