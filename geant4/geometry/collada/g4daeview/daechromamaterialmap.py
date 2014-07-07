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

    def make_code2shortname(self, code2name):
        code2short = {}
        for code, name in code2name.items():
            code2short[code] = name[:10]
        return code2short

    def _get_code2shortname(self):
        if self._code2shortname is None:
            self._code2shortname = self.make_code2shortname(self.code2name)
        return self._code2shortname
    code2shortname = property(_get_code2shortname)

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

