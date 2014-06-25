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

    NB returning None here prevents the mask or bits matching
    from being applied
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
    Simple holder of the parameters of photon presentation, that 
    could potentially be changed during operation 
    via uniforms/constants.
 
    #. initially set from config defaults or commandline options
    #. updated by menus, external messages etc..   


    """
    def __init__(self, config):
        self._mask = config.args.mask
        self._bits = config.args.bits
        self._time = 0
        self.pid  = config.args.pid
        self.fpholine = config.args.fpholine
        self.fphopoint = config.args.fphopoint
        self.debugshader = config.args.debugshader
        self.shadermode = config.args.shadermode
        self.prescale = config.args.prescale
        self.max_slots = config.args.max_slots
        #self.observers = []

    reconfigurables = ['fpholine','fphopoint','mask','bits','shadermode', 'time','pid',]

    def reconfig(self, conf):        
        update = False
        for k, v in conf:
            if k in self.reconfigurables:
                setattr(self, k, v ) 
                update = True
            else:
                log.info("ignoring %s %s " % (k,v))
            pass 
        return update

    def _get_shader_fparam(self):
        return [self.fpholine, self.fphopoint, 0., 0.]
    shader_fparam = property(_get_shader_fparam)

    def _get_shader_iparam(self):
        """
        #. cannot pass None into a shader uniform 
        """
        mask = self.mask
        bits = self.bits
        pid = self.pid
        mask = -1 if mask is None else mask 
        bits = -1 if bits is None else bits 
        pid  = -1 if pid is None else pid
        return [mask, bits, pid, 0]
    shader_iparam = property(_get_shader_iparam)

    def _get_kernel_mask(self):
        mask = self.mask
        bits = self.bits
        pid = self.pid
        mask = -1 if mask is None else mask 
        bits = -1 if bits is None else bits 
        pid  = -1 if pid is None else pid 
        return [mask, bits, pid, 0]
    kernel_mask = property(_get_kernel_mask)

    def _get_mask(self):
        return arg2mask(self._mask)
    def _set_mask(self, mask):
        self._mask = mask
    mask = property(_get_mask, _set_mask)

    def _get_bits(self):
        return arg2mask(self._bits)
    def _set_bits(self, bits):
        self._bits = bits
    bits = property(_get_bits, _set_bits )

    def _get_pid(self):
        return self._pid
    def _set_pid(self, pid):
        self._pid = int(pid)
    pid = property(_get_pid, _set_pid )




if __name__ == '__main__':
    pass


