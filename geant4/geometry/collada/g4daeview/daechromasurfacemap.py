#!/usr/bin/env python
import os, logging, json
log = logging.getLogger(__name__)

from daechromamap import DAEChromaMap

class DAEChromaSurfaceMap(DAEChromaMap):
    path = property(lambda self:self.config.chroma_surface_map)

    prefixes = ['__dd__Geometry__AdDetails__AdSurfacesAll__',
                '__dd__Geometry__PoolDetails__PoolSurfacesAll__',
                '__dd__Geometry__PoolDetails__NearPoolSurfaces__',]

    postfix = 'Surface'

    def __init__(self, config, csm=None ):
        """
        :param csm: dict of surface names keyed by integer codes
        """
        log.info("DAEChromaSurfaceMap")
        DAEChromaMap.__init__(self, config, csm )

    def shorten(self, name):
        """
        name shortener
        """
        for prefix in self.prefixes:
            if name.startswith(prefix) and name.endswith(self.postfix):
                return name[len(prefix):-len(self.postfix)]
            pass
        pass
        return name

if __name__ == '__main__':
    from daeconfig import DAEConfig
    config = DAEConfig()
    config.init_parse()

    csm = DAEChromaSurfaceMap.fromjson(config)
    print csm

    names_in = "RSOil,UnistrutRib4"
    codes = csm.convert_names2codes(names_in)
    names = csm.convert_codes2names(codes)
    print "codes %s names %s " % (codes, names)





