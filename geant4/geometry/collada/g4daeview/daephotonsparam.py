#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)


try:
    from chroma.event import Photons, mask2arg_, arg2mask_, PHOTON_FLAGS
except ImportError:
    from photons import Photons, mask2arg_, arg2mask_, PHOTON_FLAGS

def arg2mask( argl ):
    """ 
    Return strings representing integers as integers
    otherwise perform enum name to mask conversion.
    """
    if argl is None or argl == "NONE":return None
    mask = 0 
    try:
        mask = int(argl)
    except ValueError:
        mask = arg2mask_(argl)
    pass
    return mask





class DAEPhotonsParam(object):
    """
    Simple holder of the parameters of photon presentation
    
    #. initially set from config defaults or commandline options
    #. updated by menus, external messages etc..   
    """
    def __init__(self, config):
        self._mask = config.args.mask
        self._bits = config.args.bits
        self.fpholine = config.args.fpholine
        self.pholine = config.args.pholine
        self.fphopoint = config.args.fphopoint
        self.phopoint = config.args.phopoint

    def _get_shader_uniform_param(self):
        return [self.fpholine, self.fphopoint, 0., 0.]
    shader_uniform_param = property(_get_shader_uniform_param)


    def _get_mask(self):
        return arg2mask(self._mask)
    def _set_mask(self, mask):
        self._mask = mask

    def _get_bits(self):
        return arg2mask(self._bits)
    def _set_bits(self, bits):
        self._bits = bits


    mask = property(_get_mask, _set_mask)
    bits = property(_get_bits, _set_bits )


    def reconfig(self, conf):
        update = False
        for k, v in conf:
            if k == 'fpholine':
                self.fpholine = v
                update = True
            elif k == 'fphopoint':
                self.fphopoint = v
            elif k == 'pholine':
                self.pholine = True   # remove
                update = True
            elif k == 'phopoint':
                self.pholine = False  # remove 
                update = True
            elif k == 'mask':
                self._mask = v
                update = True
            else:
                log.info("ignoring %s %s " % (k,v))
            pass 
        return update


