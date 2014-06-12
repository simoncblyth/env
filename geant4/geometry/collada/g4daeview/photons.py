#!/usr/bin/env python
"""
**WARNING** : The class `photons_fallback.Photons` duplicates `chroma.event.Photons` (in my forked chroma)

Duplication as wish to provide a fallback for installations without Chroma

"""
import logging
log = logging.getLogger(__name__)

try:
    from chroma.event import Photons
    log.info("using chroma.event.Photons")
except ImportError:
    from photons_fallback import Photons
    log.info("using photons_fallback.Photons")






