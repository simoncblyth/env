#!/usr/bin/env python
import os, logging, json
log = logging.getLogger(__name__)

from daechromamap import DAEChromaMap

class DAEChromaMaterialMap(DAEChromaMap):
    path = property(lambda self:self.config.chroma_material_map)

    prefix = "__dd__Materials__"

    def __init__(self, config, cmm=None ):
        """
        :param cmm: dict of short material names keyed by integer codes
        """
        DAEChromaMap.__init__(self, config, cmm )

    def shorten(self, name):
        if name.startswith(self.prefix) and name[-9:-7] == '0x': 
            return name[len(self.prefix):-9]
        return name

if __name__ == '__main__':
    from daeconfig import DAEConfig
    config = DAEConfig()
    config.init_parse()
    cmm = DAEChromaMaterialMap.fromjson(config)
    print cmm.path
    print cmm

    names_in = "Acrylic,GdDopedLS"
    codes = cmm.convert_names2codes(names_in)
    names = cmm.convert_codes2names(codes)
    print "codes %s names %s " % (codes, names)







