#!/usr/bin/env python
"""
For checking using LXe example::

    g4daeview.sh -p lxe -g 1: -n1

    lxe-
    lxe-test   # sling some photons via RootZMQ


Classes `chroma.event.Photons` and `photons.Photons` 
should be identical, doing fallback import as want to support installations
without chroma

TODO: test without chroma running

NO_HIT                                     1   0x1      
BULK_ABSORB                                2   0x2      
SURFACE_DETECT                             4   0x4      
SURFACE_ABSORB                             8   0x8      
RAYLEIGH_SCATTER                          16   0x10      
REFLECT_DIFFUSE                           32   0x20      
REFLECT_SPECULAR                          64   0x40      
SURFACE_REEMIT                           128   0x80      
SURFACE_TRANSMIT                         256   0x100      
BULK_REEMIT                              512   0x200      
NAN_ABORT                         2147483648   0x80000000      

"""
import logging
import numpy as np
log = logging.getLogger(__name__)


import OpenGL.GL as gl
import OpenGL.GLUT as glut


DRAWMODE = { 'lines':gl.GL_LINES, 'points':gl.GL_POINTS, }

from env.graphics.color.wav2RGB import wav2RGB
try:
    from chroma.event import Photons, mask2arg_, arg2mask_, PHOTON_FLAGS
except ImportError:
    from photons import Photons, mask2arg_, arg2mask_, PHOTON_FLAGS

from daegeometry import DAEMesh 
from daemenu import DAEMenu
from daevertexbuffer import DAEVertexBuffer

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


   

class DAEPhotons(object):
    """
    A wrapper around the underlying `photons` instance that 
    handles presentation.
    """ 
    def __init__(self, photons, event, pmtid=None ):

        self.invalidate_photons()
        self._photons = photons  
        pass
        self.event = event        # for access to qcut
        config = event.config
        self.config = config
        self.pmtid = pmtid        # unused
        pass
        if not event is None:
            self.reconfig([
                       ['fpholine', config.args.fpholine],
                       ['pholine',  config.args.pholine],
                       ['fphopoint',config.args.fphopoint],
                       ['phopoint', config.args.phopoint],
                      ])
        pass
        self.update_flags_menu()


    def __repr__(self):
        return "%s %s " % (self.__class__.__name__, self.nphotons)

    nphotons = property(lambda self:len(self._photons))
    vertices = property(lambda self:self._photons.pos)   # allows to be treated like DAEMesh 
    momdir = property(lambda self:self._photons.dir)

    @classmethod
    def make_menutree( cls ):
        """
        Having menus coming and going is problematic, so create tree of placeholder submenus
        """ 
        photons = DAEMenu("photons")
        flags = DAEMenu("flags")
        history = DAEMenu("history")
        photons.addSubMenu(flags)
        photons.addSubMenu(history) 
        return photons 

    def update_flags_menu(self):
        """
        Only needs to be called once at instantiation to populate the placeholder menu 
        """
        log.info("update_flags_menu")
        flags_menu = self.config.rmenu.find_submenu("flags")
        flags_menu.addnew("ANY", self.flags_callback )
        for name in sorted(PHOTON_FLAGS, key=lambda _:PHOTON_FLAGS[_]):
            log.info("update_flags_menu %s " % name )
            flags_menu.addnew(name, self.flags_callback )
        pass
        flags_menu.update()

    def update_history_menu(self, photons  ):
        history_menu = self.config.rmenu.find_submenu("history")
        assert history_menu

        nflag, history = photons.history() 
        log.info("change_history_menu : nflag %s unique flag combinations len(history) %s " % (nflag, len(history)))

        history_menu.addnew( "ANY", self.history_callback, mask=None )
        for mask,count in sorted(history,key=lambda _:_[1], reverse=True):
            frac = float(count)/nflag
            title = "[0x%x] %d (%5.2f): %s " % (mask, count, frac, mask2arg_(mask)) 
            history_menu.addnew( title, self.history_callback, mask=mask )
        pass
        history_menu.update()
 
    def history_callback(self, item):
        self.config.args.mask = None
        self.config.args.bits = item.extra['mask']  
        self.invalidate_vbo()
        self.config.rmenu.dispatch('on_needs_redraw')

    def flags_callback(self, item ):
        name = item.title
        allowed = PHOTON_FLAGS.keys() + ['ANY']
        assert name in allowed, name
        log.info("flags_callback setting config.args.mask to %s " % name )
        if name == 'ANY':name = 'NONE'
        self.config.args.mask = name 
        self.config.args.bits = None 
        self.invalidate_vbo()
        self.config.rmenu.dispatch('on_needs_redraw')

    def invalidate_photons(self):
        """
        When changing photons everything is invalidated
        """ 
        self._photons = None
        self._mesh = None
        self._color = None
        self._mode = None
        self._ldata = None   
        self._pdata = None   
        self._drawmode = None
        self._lvbo = None   
        self._pvbo = None   
        self._photon_indices = None

    def invalidate_vbo(self):
        """
        When reconfiguring presentation just these are invalidated
        """
        log.info("invalidate_vbo")
        self._mode = None
        self._ldata = None   
        self._pdata = None   
        self._drawmode = None
        self._lvbo = None   
        self._pvbo = None   
        self._photon_indices = None

    def _set_photons(self, photons):
        """
        photons setter invalidates everything
        """
        self.invalidate_photons()
        self._photons = photons
    def _get_photons(self):
        return self._photons    
    photons = property(_get_photons, _set_photons)


    def _get_color(self):
        if self._color is None:
            self._color = self.wavelengths2rgb()
        return self._color
    color = property(_get_color)

    def _get_pdata(self):
        if self._pdata is None:
            self._pdata = self.create_pdata()
        return self._pdata
    pdata = property(_get_pdata)         

    def _get_ldata(self):
        if self._ldata is None:
            self._ldata = self.create_ldata()
        return self._ldata
    ldata = property(_get_ldata)         

    def _get_pvbo(self):
        if self._pvbo is None:
           self._pvbo = self.create_vbo(self.pdata)  
        return self._pvbo
    pvbo = property(_get_pvbo)  

    def _get_lvbo(self):
        if self._lvbo is None:
           self._lvbo = self.create_vbo(self.ldata)  
        return self._lvbo
    lvbo = property(_get_lvbo)  

    def _get_mesh(self):
        if self._mesh is None:
            self._mesh = DAEMesh(self.vertices)
        return self._mesh
    mesh = property(_get_mesh)

    def _get_mode(self):
        if self._mode is None:
            self._mode = 'lines' if self.config.args.pholine else 'points'
        return self._mode
    mode = property(_get_mode)

    def _get_drawmode(self):
        if self._drawmode is None:
            self._drawmode = DRAWMODE[self.mode]
        return self._drawmode
    drawmode = property(_get_drawmode)

    def _get_photon_indices(self):
        if self._photon_indices is None:
            self._photon_indices = self.select_photon_indices()
        return self._photon_indices 
    photon_indices = property(_get_photon_indices)

    def wavelengths2rgb(self):
        color = np.zeros(self.nphotons, dtype=(np.float32, 4))
        for i,wl in enumerate(self.photons.wavelengths):
            color[i] = wav2RGB(wl)
        return color

    def create_pdata(self):
        data = np.zeros(self.nphotons, [('position', np.float32, 3), 
                                        ('color',    np.float32, 4)]) 
        data['position'] = self.vertices
        data['color']    = self.color
        return data

    def create_ldata(self):
        """
        #. interleave the photon positions with sum of photon position and direction
        #. interleaved double up the colors 
        """
        data = np.zeros(2*self.nphotons, [('position', np.float32, 3), 
                                          ('color',    np.float32, 4)]) 

        vertices = np.empty((len(self.vertices)*2,3), dtype=self.vertices.dtype )
        vertices[0::2] = self.vertices
        vertices[1::2] = self.vertices + self.momdir*self.config.args.fpholine

        colors = np.empty((len(self.color)*2,4), dtype=self.color.dtype )
        colors[0::2] = self.color
        colors[1::2] = self.color

        data['position'] = vertices
        data['color']    = colors
        return data 



    def select_photon_indices(self):
        """
        Use the photon history flags (array of bit fields)
        to make bit mask and history selections
        """
        npho = self.nphotons
        flags = self.photons.flags   
        nflag = len(flags)
        assert nflag == npho, (nflg, npho)

        self.update_history_menu( self.photons )

        argm = self.config.args.mask 
        argb = self.config.args.bits 

        mask = arg2mask(argm)
        bits = arg2mask(argb)

        log.debug("argm %s mask %s " % (argm, mask))
        log.debug("argb %s bits %s " % (argm, bits))
        log.info("photon_indices npho %s nflag %s mask %s bits %s " % (npho,nflag,mask,bits))

        if mask is None and bits is None:
            pindices = np.arange( npho, dtype=np.uint32)  
        elif not mask is None:
            pindices = np.where( flags & mask )[0]    # bitwise AND for flags selection
        elif not bits is None:
            pindices = np.where( flags == bits )[0]   # equality for history bits selection
        else:
            assert 0

        log.info("photon_indices selection : %s " % len(pindices) )
 
        return pindices

    def create_vbo(self, data):
        """
        #. when no photons are selected, a token single indice is used that avoids a crash
        #. for the points pvbo its 1-1 between vertices and photons
        #. for the lines lvbo is 2-1
        """
        nvtx = data.size
        npho = self.nphotons
        assert nvtx == npho or nvtx == 2*npho, (nvtx,npho)
        pindices = self.photon_indices

        if len(pindices) == 0:
            vindices = np.arange( 1, dtype=np.uint32 ) 
        else:
            if nvtx == npho:
                vindices = pindices
            elif nvtx == npho*2:
                vindices = np.empty(2*len(pindices), dtype=np.uint32)
                vindices[0::2] = 2*pindices
                vindices[1::2] = 2*pindices + 1
                log.info(" lvbo vindices %s " % len(vindices))
            else:
                assert 0, (nvtx,npho)
            pass
        pass
        return DAEVertexBuffer( data, vindices  )

    def draw(self):
        """
        qcut restricts elements drawn, the default of 1 corresponds to all

        Note that the VBO vertices are duplicated once for the line and once for
        the points, presumably there is some clever way to control the strides to
        avoid that ?

        Changing presentation must account for the OpenGL state when this
        gets called from daeframeghandler
        """ 
        qcount = int(len(self.pdata)*self.event.qcut)
        self.lvbo.draw(mode=gl.GL_LINES,  what='pc', count=2*qcount, offset=0 )
        self.pvbo.draw(mode=gl.GL_POINTS, what='pc', count=qcount  , offset=0 )


    def reconfig(self, conf):
        update = False
        for k, v in conf:
            if k == 'fpholine':
                self.config.args.fpholine = v
                update = True
            elif k == 'fphopoint':
                self.config.args.fphopoint = v
                gl.glPointSize(self.config.args.fphopoint)
            elif k == 'pholine':
                self.config.args.pholine = True
                update = True
            elif k == 'phopoint':
                self.config.args.pholine = False
                update = True
            elif k == 'mask':
                self.config.args.mask = v
                update = True
            else:
                log.info("ignoring %s %s " % (k,v))
            pass 
        pass
        if update:
            self.invalidate_vbo()

    def draw_extremely_slowly(self):
        """
        """
        gl.glDisable( gl.GL_LIGHTING )
        gl.glDisable( gl.GL_DEPTH_TEST )

        for i in range(self.nphotons):
            a = self.vertices[i]
            b = a + self.momdir[i]*100

            gl.glBegin( gl.GL_LINES)
            gl.glVertex3f( *a )
            gl.glVertex3f( *b )
            gl.glEnd()
        pass 

        gl.glEnable( gl.GL_LIGHTING )
        gl.glEnable( gl.GL_DEPTH_TEST )




if __name__ == '__main__':
    pass




