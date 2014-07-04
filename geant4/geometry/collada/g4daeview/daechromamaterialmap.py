#!/usr/bin/env python
import os, logging, json
log = logging.getLogger(__name__)

class DAEChromaMaterialMap(object):
    name = "chroma_material_map.json"
    path = property(lambda self:self.config.resolve_confpath(self.name))

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

    def __init__(self, config, cmm=None ):
        """
        :param cmm: dict of short material names keyed by integer codes
        """
        log.info("DAEChromaMaterialMap")
        self.config = config
        self.code2name = cmm

    def dump(self):
        log.info("dump")
        print self

    def __str__(self):
        d = self.code2name 
        return "\n".join(["%3d %s " % (k, d[k]) for k in sorted(d.keys(),key=lambda _:_)])

    def write(self):
        if os.path.exists(self.path):
            prior = DAEChromaMaterialMap.fromjson(self.config) 
            mismatch = DAEChromaMaterialMap.compare( prior, self )
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
    from daeconfig import DAEConfig
    config = DAEConfig()
    config.init_parse()
    cmm = DAEChromaMaterialMap.fromjson(config)
    print cmm

