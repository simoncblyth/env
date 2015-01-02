#!/usr/bin/env python
"""
DAEPhotonsStyler
==================

DAEPhotonsStyler Usage by DAEPhotons together with DAEPhotonsRenderer
-----------------------------------------------------------------------------

#. DAEPhotonsStyler used together with DAEPhotonsRenderer as constituents of DAEPhotons
#. cfglist property of styler steers the renderer in DAEPhotons.draw

::

    084         self.styler = DAEPhotonsStyler()
    ...
    090         self.renderer = DAEPhotonsRenderer(self, event.scene.chroma ) # pass chroma context to renderer for PyCUDA/OpenGL interop tasks 
    ...
    180     def _get_cfglist(self):
    181         return self.styler.get_list(self.style)
    182     cfglist = property(_get_cfglist)
    ...
    276     def draw(self):
    ...
    281         if self.photons is None:return
    282         self.renderer.update_constants()
    283         for cfg in self.cfglist:
    284             self.drawcfg( cfg )
    285 
    286     def drawcfg(self, cfg ):
    287         self.renderer.shaderkey = cfg['shaderkey']
    288         if cfg['drawkey'] == 'multidraw':
    289             counts, firsts, drawcount = self.analyzer.counts_firsts_drawcount
    290             self.renderer.multidraw(mode=cfg['drawmode'],slot=cfg['slot'],
    291                                       counts=counts,
    292                                       firsts=firsts,
    293                                    drawcount=drawcount, extrakey=cfg['extrakey'] )
    294         else:
    295             self.renderer.draw(mode=cfg['drawmode'],slot=cfg['slot'])


Styles
--------

============  ===========      ============  ============  ==========  ========   ==============
style          drawmode         drawkey       shaderkey     extrakey    slot       description 
============  ===========      ============  ============  ==========  ========   ==============
noodles        GL_POINTS        multidraw      p2l           None       None       all steps  
movie          GL_POINTS        multidraw      p2l           None       -1         
movie-extra    GL_POINTS        multidraw      p2l           p2p        -1          
dmovie         GL_POINTS        draw           p2l           p2p        -1         simple animation slot draw, expected to be efficient
confetti       GL_POINTS        multidraw      nogeo         None       None       points for every step   
confetti-0     GL_POINTS        multidraw      nogeo         None        0         first slot points
confetti-2     GL_POINTS        multidraw      nogeo         None       -2         last slot points
confetti-1     GL_POINTS        multidraw      nogeo         None       -1         animation slot points
dconfetti-1    GL_POINTS        draw           nogeo         None       -1         animation slot points, expected to be efficient
spagetti       GL_LINE_STRIP    multidraw      nogeo         None       None       trajectory lines for each photon 
============  ===========      ============  ============  ==========  ========   ==============


Ingredients of Style
-----------------------

*slot*

   #. -1, animation top slot (max_slots-1) 

      content is calculated by presenter CUDA kernel 
      by interpolation of relevant slot pair straddling 
      the time parameter input 

   #. None, 

      corresponds to using a max_slots value of 1 and slot 0, 
      ie see all slots, but exclude slot -1 (?how?)

      How does this handle empty slots and exclusion of slot -1  

      It doesnt see them, as it uses **multidraw** 
      which has a number of filled slots input array 
      to the multidraw call 

*drawkey*

   #. `multidraw` is way of in effect doing separate draw calls 
       for each photon (or photon history) eg allowing trajectory line presentation.

       It is so prevalent as without it have no choice but to 
       restrict to slots that will always be present, ie slot 0 and slot -1 and -2
       (unless traversed the entire VBO with selection to skip empties ?)



"""
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
    style_names = ['noodles','movie','movie-extra','dmovie','spagetti','confetti','confetti-0','confetti-2','confetti-1','dconfetti-1','genstep',]
    style_names_menu = property(lambda self:self.style_names + ["spagetti,confetti","noodles,confetti"] )

    def __init__(self):
        self.styles = self._make_styles(self.style_names)

    def _make_styles(self, style_names):
        """
        :param style_names: list of names 
        :return: dict of cfg dicts keyed by style name
        """
        styles = {}
        for name in style_names:
            styles[name] = self._make_cfg(name) 
        pass
        return styles

    def get_list(self, style):
        """
        :param style: comma delimited list of style names eg "noodles,confetti"
        :return: list of style dicts 
        """
        cfgs = []
        for name in style.split(","):
            cfg = self._get(name)
            if cfg is None:
                log.warn("no such style %s " % name )
            else:
                cfgs.append(cfg)
            pass
        return cfgs 

    def _get(self, style):
        """
        :param style: name 
        :return: style dict or None
        """
        return self.styles.get(style,None)

    def _make_cfg(self, style):
        """
        :param style: string identifying various techniques to present the photon information

:e
        Debug tips:

        #. check time dependancy with `udp.py --time 10` etc..

        Animated spagetti, ie LINE_STRIP that animates: not easy 
        as need multidraw dynamic counts with interpolated top slot 
        interposition. Technically challenging but not so informative.
        Would be tractable is could get geometry shader to deal in LINE_STRIPs.

        A point representing ABSORPTIONs would be more useful.


        (FIXED) Shader Switching Issue 
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~         

        Live transitions to the "nogeo" shaders `spagetti` 
        and `confetti` are working from all others.  
        The reverse transitions from "nogeo" to "point2line" 
        shaderkey do not work, giving a blank render.

        Presumably an attribute binding problem, not changing a part 
        of opengl state 

        Fixed by keeping shaders alive and never deleting 
        them just switch between them.

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

        elif style == 'genstep':

           cfg['description'] = "genstep dev" 
           cfg['drawmode'] = gl.GL_POINTS
           cfg['drawkey'] = "draw" 
           cfg['shaderkey'] = "nogeo"
           cfg['slot'] = None

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


