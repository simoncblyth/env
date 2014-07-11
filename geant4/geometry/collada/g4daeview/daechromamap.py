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
        uk = set(a.keys()).union(set(b.keys()))

        if len(a) != len(b):
            log.warn("KEY MISMATCH a %s b %s union %s  " % (repr(a.keys()), repr(b.keys()), repr(uk) ))

        mismatch = 0 
        for k in uk:
            ak = a.get(k,None)
            bk = b.get(k,None)
            same = ak == bk
            mkr = "" if same else "********"
            if not same:mismatch += 1
            print "%2d %-20s %-20s %s" % ( k, ak, bk, mkr )
        pass
        log.info("compare sees %s mismatches " % mismatch )
        return mismatch

    def dump(self):
        log.info("dump")
        print self

    def __init__(self, config, c2n=None):
        self.config = config
        self.code2name = c2n
        self._code2shortname = None
        self._shortname2code = None
        self._name2code = None

    def __str__(self):
        hdr = "%s with %s code2name keys " % (self.__class__.__name__, len(self.code2name))
        d = self.code2name 
        s = self.code2shortname 
        return "\n".join([hdr] + ["%3d %25s    %s " % (k, s[k], d[k]) for k in sorted(d.keys(),key=lambda _:_)])

    def write(self):
        if os.path.exists(self.path):
            if self.config.args.wipegeometry:
                log.info("unlinking %s due to --wipegeometry " % self.path)
                os.unlink(self.path) 
            else:
                prior = self.fromjson(self.config) 
                mismatch = self.compare( prior, self )
            pass
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


    def make_name2code(self, code2name):
        return dict((name,code) for code,name in code2name.items())
    def make_code2shortname(self, code2name):
        return dict((code, self.shorten(name)) for code, name in code2name.items())
    def make_shortname2code(self, code2name):
        return dict((self.shorten(name),code) for code, name in code2name.items())

    def _get_code2shortname(self):
        if self._code2shortname is None:
            self._code2shortname = self.make_code2shortname(self.code2name)
        return self._code2shortname
    def _get_shortname2code(self):
        if self._shortname2code is None:
            self._shortname2code = self.make_shortname2code(self.code2name)
        return self._shortname2code
    def _get_name2code(self):
        if self._name2code is None:
            self._name2code = self.make_name2code(self.code2name)
        return self._name2code

    code2shortname = property(_get_code2shortname)
    shortname2code = property(_get_shortname2code)
    name2code = property(_get_name2code)


    def convert_names2codes(self,names, short=True):
        """
        Comma delimited string of short names -> list of integer codes
        """
        d = self.shortname2code if short else self.name2code
        codes = map( lambda _:d.get(_), names.split(",") )  
        assert len(codes) <=4, codes 
        default = [-1,-1,-1,-1]
        codes = codes + default[:4-len(codes)] 
        return codes

    def convert_codes2names(self, codes, short=True):
        """
        List of integer codes -> comma delimited string of short material names
        """
        d = self.code2shortname if short else self.code2name
        names = map( lambda _:d.get(_,None), codes )
        return ",".join(filter(None,names))  

    def code2str(self, code, short=True):
        d = self.code2shortname if short else self.code2name
        return d.get(code,"-")

    def paircode2str(self, paircode):
        c2s_ = lambda c:self.code2str(c,short=True)
        return ",".join(map(c2s_,[paircode//1000,paircode%1000]))

    def str2paircode(self, names):
        codes = self.convert_names2codes(names)
        return codes[0]*1000 + codes[1]





if __name__ == '__main__':
    pass



