#!/usr/bin/env python

import os, collada, logging
log = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    dae = collada.Collada("subcopy.dae")
    print dae

    boundgeom = list(dae.scene.objects('geometry'))
    print boundgeom



