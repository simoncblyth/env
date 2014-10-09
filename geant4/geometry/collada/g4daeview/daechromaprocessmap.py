#!/usr/bin/env python
import os, logging, json
log = logging.getLogger(__name__)

from daechromamap import DAEChromaMap

class DAEChromaProcessMap(DAEChromaMap):
    """
    NB by not importing chroma directly, instead taking control of the 
    dict via an argument and persisting it into json this opens 
    gateway for the non-chroma aware to deal in chroma process codes.
    """
    path = property(lambda self:self.config.chroma_process_map)

    def __init__(self, config, cpm=None ):
        """
        :param cmm: dict of short material names keyed by integer codes
        """
        DAEChromaMap.__init__(self, config, cpm )
        self._code2shortname = None

    def make_code2shortname(self, code2name):
        code2short = {}
        for code, name in code2name.items():
            elem = name.split("_")
            #code2short[code] = "%s_%s" % ( elem[0][0], elem[1][:3]) 
            code2short[code] = "%s_%s" % ( elem[0][0], elem[1] ) 
        return code2short

    def _get_code2shortname(self):
        if self._code2shortname is None:
            self._code2shortname = self.make_code2shortname(self.code2name)
        return self._code2shortname
    code2shortname = property(_get_code2shortname)

    def mask2str(self, mask, short=True):
        """
        cpm.mask2str(1+2+4+8) => NO_HIT,BULK_ABSORB,SURFACE_DETECT,SURFACE_ABSORB
        """ 
        d = self.code2shortname if short else self.code2name
        return ",".join(map(lambda _:_[1],filter(lambda _:_[0] & mask, sorted(d.items(),key=lambda _:_[0]))))


if __name__ == '__main__':
    from daeconfig import DAEConfig
    config = DAEConfig()
    config.init_parse()
    cpm = DAEChromaProcessMap.fromjson(config)
    print cpm

    print cpm.mask2str(1+2+4+8) 



