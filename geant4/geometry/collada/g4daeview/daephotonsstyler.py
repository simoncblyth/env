#!/usr/bin/env python

import logging, pprint
log = logging.getLogger(__name__)

import OpenGL.GL as gl
import numpy as np


class DAEPhotonsStyler(object):
    """
    High level photon rendering configuration, controlled via 
    DAEPhotons.style property, which dictates contents of 
    DAEPhotons.cfglist property.

    The style can be changed live via menu or UDP::

       udp.py --style noodles,movie

    """
    style_names = ['noodles','movie','movie-extra','dmovie','spagetti','confetti','confetti-0','confetti-2','confetti-1','dconfetti-1',]

    def __init__(self):
        self.styles = self._make_styles(self.style_names)

    def _make_styles(self, style_names):
        styles = {}
        for name in style_names:
            styles[name] = self._make_cfg(name) 
        pass
        return styles

    def get_list(self, style):
        """
        :param style: comma delimited list of style names
        :return: list of style dicts 
        """
        cfgs = []
        for name in style.split(","):
            cfg = self.get(name)
            if cfg is None:
                log.warn("no such style %s " % name )
            else:
                cfgs.append(cfg)
            pass
        return cfgs 

    def get(self, style):
        """
        :param style: name 
        :return: style dict or None
        """
        return self.styles.get(style,None)

    def _make_cfg(self, style):
        """
        :param photonskey: string identifying various techniques to present the photon information

        *slot*

           #. -1, top slot at max_slots-1
           #. None, corresponds to using max_slots 1 with slot 0,
              with top slot excluded 
              (ie seeing all steps of the propagation except the artificial 
              interpolated top slot)

        *drawkey*
           `multidraw` is efficient way of in effect doing separate draw calls 
           for each photon (or photon history) eg allowing trajectory line presentation.

           It is so prevalent as without it have no choice but to 
           restrict to slots that will always be present, ie slot 0 and slot -1.
           (unless traversed the entire VBO with selection to skip empties ?)

        Debug tips:

        #. check time dependancy with `udp.py --time 10` etc..

        Animated spagetti, ie LINE_STRIP that animates: not easy 
        as need multidraw dynamic counts with interpolated top slot 
        interposition. Technically challenging but not so informative.
        Would be tractable is could get geometry shader to deal in LINE_STRIPs.

        A point representing ABSORPTIONs would be more useful.


        Live transitions to the "nogeo" shaders `spagetti` 
        and `confetti` are working from all others.  
        The reverse transitions from "nogeo" to "point2line" 
        shaderkey do not work, giving a blank render.

        Presumably an attribute binding problem, not changing a part 
        of opengl state 


        Slot0 Selection Immunity Issue
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

        Confetti styles were immune to pid/mask/history selection, 
        until added `init_ccol.w  = skip_alpha` in propagate_vbo.cu:present_vbo


        Color By Flag Change
        ~~~~~~~~~~~~~~~~~~~~~


        """
        cfg = {}
        cfg['extrakey'] = None

        if style == 'noodles':

           cfg['description'] = "multidraw POINTS geometry shader pulled into photon direction direction at each step of the photon" 
           cfg['drawmode'] = gl.GL_POINTS
           cfg['drawkey'] = "multidraw" 
           cfg['shaderkey'] = "p2l"
           cfg['extrakey'] = None   # switching to p2p causes to become like confetti ?
           cfg['slot'] = None  

        elif style == 'movie':

           cfg['description'] = "LINE_STRIP direction/polarization that is time interpolated " 
           cfg['drawmode'] = gl.GL_POINTS
           cfg['drawkey'] = "multidraw" 
           cfg['shaderkey'] = "p2l"
           cfg['slot'] = -1  

        elif style == 'movie-extra':

           cfg['description'] = "LINE_STRIP direction/polarization that is time interpolated " 
           cfg['drawmode'] = gl.GL_POINTS
           cfg['drawkey'] = "multidraw" 
           cfg['shaderkey'] = "p2l"
           cfg['extrakey'] = "p2p"   
           cfg['slot'] = -1    

        elif style == 'dmovie':

           cfg['description'] = "simple animation slot draw for efficiency" 
           cfg['drawmode'] = gl.GL_POINTS
           cfg['drawkey'] = "draw" 
           cfg['shaderkey'] = "p2l"
           cfg['extrakey'] = "p2p" 
           cfg['slot'] = -1    

        elif style == 'confetti':

           # confetti styles only differ in the slots 

           cfg['description'] = "POINTS for every step of the photon" 
           cfg['drawmode'] = gl.GL_POINTS
           cfg['drawkey'] = "multidraw" 
           cfg['shaderkey'] = "nogeo"
           cfg['slot'] = None

        elif style == 'confetti-2':

           cfg['description'] = "last slot POINTS" 
           cfg['drawmode'] = gl.GL_POINTS
           cfg['drawkey'] = "multidraw" 
           cfg['shaderkey'] = "nogeo"
           cfg['slot'] = -2

        elif style == 'confetti-1':

           cfg['description'] = "top slot POINTS" 
           cfg['drawmode'] = gl.GL_POINTS
           cfg['drawkey'] = "multidraw" 
           cfg['shaderkey'] = "nogeo"
           cfg['slot'] = -1

        elif style == 'dconfetti-1':

           cfg['description'] = "top slot POINTS" 
           cfg['drawmode'] = gl.GL_POINTS
           cfg['drawkey'] = "draw" 
           cfg['shaderkey'] = "nogeo"
           cfg['slot'] = -1

        elif style == 'confetti-0':

           cfg['description'] = "first slot POINTS" 
           cfg['drawmode'] = gl.GL_POINTS
           cfg['drawkey'] = "multidraw" 
           cfg['shaderkey'] = "nogeo"
           cfg['slot'] = 0

        elif style == 'spagetti':

           # time dependant slot selection `--mode 0` makes a mess 
           # for spagetti style, its kinda incompatible as current
           # skip the point works by shooting the point off to infinity ?

           cfg['description'] = "LINE_STRIP trajectory of each photon, " 
           cfg['drawmode'] = gl.GL_LINE_STRIP
           cfg['drawkey'] = "multidraw" 
           cfg['shaderkey'] = "nogeo"
           cfg['slot'] = None

        else:
            assert 0, style


        return cfg 


if __name__ == '__main__':
    pass


