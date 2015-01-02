#!/usr/bin/env python
"""
"""
import logging
log = logging.getLogger(__name__)

# these names need to be changed to remove the "photons"
from daephotonsmenuctrl import DAEPhotonsMenuController
from daephotonsparam import DAEPhotonsParam
from daephotonsrenderer import DAEPhotonsRenderer
from daephotonsstyler import DAEPhotonsStyler

class DAEDrawable(object):
    animate = True
    def __init__(self, _array, event ):
        self.event = event

        geometry = event.scene.geometry
        for att in "chroma_material_map chroma_surface_map chroma_process_map".split():
            if hasattr(geometry,att):
                setattr(self,att,getattr(geometry,att))
            pass
        pass

        self.config = event.config      
        self.interop = not event.scene.chroma.dummy

        self.param = DAEPhotonsParam( event.config)
        self.renderer = DAEPhotonsRenderer(self, event.scene.chroma, event.config ) # pass chroma context to renderer for PyCUDA/OpenGL interop tasks 
        self._style = event.config.args.style  # eg noodles, confetti, spagetti
        self.styler = DAEPhotonsStyler()

        self.menuctrl = DAEPhotonsMenuController( event.config.rmenu, self.param )
        self.material = event.config.args.material    
        self.surface  = event.config.args.surface
        self.mode = event.config.args.mode    

        presenter = self.renderer.presenter
        if not presenter is None:
            presenter.time   = event.config.args.time
            presenter.cohort = event.config.args.cohort
        pass

        self._array = None
        self.array = _array

    def _get_array(self):
        return self._array 
    def _set_array(self, _array):
        log.info("_set_array")
        if _array is None:
            self._array = None
        else:
            self._array = self.handle_array(_array)
            self.post_handle_array()
        pass
    pass
    array = property(_get_array, _set_array ) 

    def handle_array(self, _array):
        assert 0, "array must be handled by subclass"

    def post_handle_array(self):
        pass

    #### DAEPhotonsStyler driven rendering  ####

    def _get_style(self):
        return self._style
    def _set_style(self, style):
        if style == self._style:return
        self._style = style   
    style = property(_get_style, _set_style, doc="Photon presentation style, eg confetti/spagetti/movie/...") 

    def _get_cfglist(self):
        return self.styler.get_list(self.style)
    cfglist = property(_get_cfglist)


    def draw(self):
        """
        multidraw mode relies on analyzing the propagated VBO to access the 
        number of propagation steps (actually filled VBO slots, as there will be truncation) 
        """
        if self.array is None:
            #log.warn("draw called by no array ")
            return
        self.renderer.update_constants()   
        for cfg in self.cfglist:
            self.drawcfg( cfg )

    def drawcfg(self, cfg ): 
        self.renderer.shaderkey = cfg['shaderkey']
        if cfg['drawkey'] == 'multidraw':
            counts, firsts, drawcount = self.analyzer.counts_firsts_drawcount 
            self.renderer.multidraw(mode=cfg['drawmode'],slot=cfg['slot'], 
                                      counts=counts, 
                                      firsts=firsts, 
                                   drawcount=drawcount, extrakey=cfg['extrakey'] )
        else:
            self.renderer.draw(mode=cfg['drawmode'],slot=cfg['slot'])


    vertices     = property(lambda self:self.array.position)  # allows to be treated like DAEMesh  

    def _get_qcount(self):
        """
        Photon count modulated by qcut which varies between 0 and 1. 
        Used for partial drawing based on a sorted quantity.
        """ 
        return int(len(self.array)*self.event.qcut)
    qcount = property(_get_qcount, doc=_get_qcount.__doc__) 

    def _get_mesh(self):
        if self._mesh is None:
            self._mesh = DAEMesh(self.array.position)
        return self._mesh
    mesh = property(_get_mesh)

    def deferred_menu_update(self):
        """
        Calling this before GLUT setup, results in duplicated menus 
        """
        if not self.interop:return
        self.menuctrl.update_style_menu( self.styler.style_names_menu, self.style_callback )

        if hasattr(self.analyzer, 'get_material_pairs'):
            self.menuctrl.update_material_menu( self.material_pairs(), self.material_callback )

    def material_pairs(self):
         return self.analyzer.get_material_pairs(self.chroma_material_map)

    def material_callback(self, item):
        matname = item.title
        matcode = item.extra['matcode']
        log.info("material_callback matname %s matindex %s  " % (matname, matcode) )
        self.material = matcode
        self.menuctrl.rootmenu.dispatch('on_needs_redraw')

    def style_callback(self, item):
        style = item.title
        self.style = style
        self.menuctrl.rootmenu.dispatch('on_needs_redraw')

    def special_callback(self, item):
        sid = int(item.extra['sid'])
        self.param.sid = sid
        self.menuctrl.rootmenu.dispatch('on_needs_redraw')


    def _set_material(self, names):
        presenter = self.renderer.presenter
        if presenter is None:
            log.warn("cannot set material selection constants when renderer.presenter is not enabled")
            return
        pass
        codes = self.chroma_material_map.convert_names2codes(names)
        log.debug("_set_mate %s => %s " % (names, codes))
        presenter.material = codes
    def _get_material(self):
        presenter = self.renderer.presenter
        if presenter is None:
            return None
        pass
        codes = presenter.material
        names = self.chroma_material_map.convert_codes2names(codes)
        log.debug("_get_mate %s => %s " % (codes, names))
        return names
    material = property(_get_material, _set_material, doc="setter copies material selection integers into GPU quad g_mate  getter returns cached value " )



    def _set_surface(self, names):
        presenter = self.renderer.presenter
        if presenter is None:
            log.warn("cannot set material selection constants when renderer.presenter is not enabled")
            return
        pass
        codes = self.chroma_surface_map.convert_names2codes(names)
        log.debug("_set_surface %s => %s " % (names, codes))
        presenter.surface = codes
    def _get_surface(self):
        presenter = self.renderer.presenter
        if presenter is None:
            return None
        pass
        codes = presenter.surface
        names = self.chroma_surface_map.convert_codes2names(codes)
        log.debug("_get_surface %s => %s " % (codes, names))
        return names
    surface = property(_get_surface, _set_surface, doc="surface: setter copies selection integers into GPU quad g_surf  getter returns cached value " )

    def _set_mode(self, mode):
        presenter = self.renderer.presenter
        if presenter is None:
            log.warn("cannot set mode selection constants when renderer.presenter is not enabled")
            return
        pass
        presenter.mode = mode
    def _get_mode(self):
        presenter = self.renderer.presenter
        if presenter is None:
            return None
        pass
        return presenter.mode
    mode = property(_get_mode, _set_mode, doc="mode: setter copies mode control integers into GPU quad g_mode  getter returns cached value " )





if __name__ == '__main__':
    pass


