#!/usr/bin/env python
import os, logging, json
log = logging.getLogger(__name__)

from daechromamap import DAEChromaMap

class DAEChromaMaterialMap(DAEChromaMap):
    path = property(lambda self:self.config.chroma_material_map)

    def __init__(self, config, cmm=None ):
        """
        :param cmm: dict of short material names keyed by integer codes
        """
        log.info("DAEChromaMaterialMap")
        DAEChromaMap.__init__(self, config, cmm )
        self._code2shortname = None
        self._name2code = None

    def make_code2shortname(self, code2name):
        code2short = {}
        for code, name in code2name.items():
            code2short[code] = name[:10]
        return code2short

    def make_name2code(self, code2name):
        return dict((name,code) for code,name in code2name.items())

    def _get_code2shortname(self):
        if self._code2shortname is None:
            self._code2shortname = self.make_code2shortname(self.code2name)
        return self._code2shortname
    code2shortname = property(_get_code2shortname)

    def _get_name2code(self):
        if self._name2code is None:
            self._name2code = self.make_name2code(self.code2name)
        return self._name2code
    name2code = property(_get_name2code)

    def convert_names2codes(self,names):
        """
        Convert a comma delimited string of short material names
        into a list of integer codes
        """
        codes = map( lambda _:self.name2code.get(_), names.split(",") )  
        assert len(codes) <=4, codes 
        default = [-1,-1,-1,-1]
        codes = codes + default[:4-len(codes)] 
        return codes

    def convert_codes2names(self, codes):
        """
        Convert list of interger codes into comma delimited string of short material names
        """
        names = map( lambda _:self.code2name.get(_,None), codes )
        return ",".join(filter(None,names))  

    def code2str(self, code, short=True):
        """
        """ 
        d = self.code2shortname if short else self.code2name
        return d.get(code,"-")

        



if __name__ == '__main__':
    from daeconfig import DAEConfig
    config = DAEConfig()
    config.init_parse()
    cmm = DAEChromaMaterialMap.fromjson(config)
    print cmm

    names_in = "Acrylic,GdDopedLS"
    codes = cmm.convert_names2codes(names_in)
    names = cmm.convert_codes2names(codes)
    print "codes %s names %s " % (codes, names)







