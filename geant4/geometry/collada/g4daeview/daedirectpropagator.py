#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)
import IPython as IP
from env.chroma.ChromaPhotonList.cpl import examine_cpl, random_cpl, save_cpl, load_cpl, create_cpl_from_photons_very_slowly
from photons import Photons

class DAEDirectPropagator(object):
    def __init__(self, config):
        self.config = config
    def propagate(self, cpl ):
        """
        :param cpl: ChromaPhotonList instance
        :return propagated_cpl: ChromaPhotonList instance
        """
        #examine_cpl( cpl ) 
        photons = Photons.from_cpl(cpl, extend=True)  # CPL into chroma.event.Photons OR photons.Photons   
        print "photons:", photons
        return cpl


def main():
    """
    Debugging CPL and Photons handling/conversions

    #. loads persisted CPL
    #. converts into `photons` chroma.event.Photons (fallback photons.Photons)
    #. creates new CPL from the `photons`

    """
    from daedirectconfig import DAEDirectConfig
    config = DAEDirectConfig(__doc__)
    config.parse()

    cpl = config.load_cpl("1")

    extend = False
    photons = Photons.from_cpl(cpl, extend=extend)  # CPL into chroma.event.Photons OR photons.Photons   

    cpl2 = create_cpl_from_photons_very_slowly(photons) 
    print cpl2

    cpl.Print()
    cpl2.Print()

    digests = (cpl.GetDigest(),cpl2.GetDigest()) 
    log.info( "digests %s " % repr(digests))

    if not extend:
        assert digests[0] == digests[1], ("Digest mismatch between cpl and cpl2 ", digests)
    else:
        assert digests[0] != digests[1], ("Digest mismatch expected in extend mode", digests)


    propagator = DAEDirectPropagator(config)
    #propagator.propagate(cpl) 

    IP.embed()

if __name__ == '__main__':
    main()


    


