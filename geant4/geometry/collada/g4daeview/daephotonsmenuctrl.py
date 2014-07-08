#!/usr/bin/env python
"""
Menu Issues
------------





FIXED : Duplicated Flags
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* resolved by defering the launch config until after GLUT setup

After loading on launch the menu photon flags are duplicated.::

   g4daeview.sh --with-chroma --load 1 

This does not happen after a launch followed by an external load::

   g4daeview.sh --with-chroma 
   udp.py --load 1
       

FIXED : History Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* resolved using analyzer.history setup

Old history selection used CPU side examination of 
chroma.event.Photon flags. Back then was doing single stepping 
pulling back to CPU and creating chroma.event.Photon at every step.

Equivalent in GPU resident approach, is to do this in the propagate 
analyzer. After running analyzer for the propagated have analyzer.history 
available


"""
import logging
log = logging.getLogger(__name__)
from daemenu import DAEMenu

try:
    from chroma.event import mask2arg_, arg2mask_, PHOTON_FLAGS
except ImportError:
    from photons import mask2arg_, arg2mask_, PHOTON_FLAGS


class DAEPhotonsMenuController(object):
    def __init__(self, rootmenu, param):
        """
        :param rootmenu: DAEMenu instance for the top menu
        :param param: menu callback receiving instance, needs mask and bits setters 
                      like DAEPhotonsParam

        """ 
        self.rootmenu = rootmenu
        self.param = param
        self.setup_menus()

    def setup_menus(self):
        """
        Having menus coming and going is problematic, so create tree of placeholder submenus
        Just structure, not content
        """
        log.debug("setup_menus")

        photons_menu = DAEMenu("photons")

        style_menu = DAEMenu("style")
        flags_menu = DAEMenu("flags")
        history_menu = DAEMenu("history")
        material_menu = DAEMenu("material")
        special_menu = DAEMenu("special")

        photons_menu.addSubMenu(style_menu)
        photons_menu.addSubMenu(flags_menu)
        photons_menu.addSubMenu(history_menu) 
        photons_menu.addSubMenu(material_menu) 
        photons_menu.addSubMenu(special_menu) 

        self.rootmenu.addSubMenu(photons_menu)

        self.photons_menu = photons_menu
        self.style_menu = style_menu
        self.flags_menu = flags_menu
        self.history_menu = history_menu
        self.material_menu = material_menu
        self.special_menu = special_menu

    def update_old(self, photons, msg=""):
        log.info("update_old photons %s %s %s" % (photons.__class__.__name__, photons, msg) )
        self.update_flags_menu()    
        self.update_history_menu_old( photons )    

    def update_propagated(self, analyzer, special_callback, msg=""):
        self.update_flags_menu()    
        self.update_history_menu( analyzer.history )    
        self.update_special_menu( analyzer.special, special_callback )    

    def update_special_menu(self, special, callback ):
        special_menu = self.rootmenu.find_submenu("special")
        assert special_menu == self.special_menu
        for sid in special:
            title = "%s" % sid
            special_menu.addnew(title, callback, sid=int(sid) )
        pass
        special_menu.update()  

    def update_style_menu(self, styles, callback ):
        style_menu = self.rootmenu.find_submenu("style")
        assert style_menu == self.style_menu
        for name in styles:
            #log.info("update_flags_menu %s " % name )
            style_menu.addnew(name, callback )
        pass
        style_menu.update()  

    def update_material_menu(self, matnamecode, callback ):
        """
        """
        material_menu = self.rootmenu.find_submenu("material")
        assert material_menu == self.material_menu
        for name, matcode in matnamecode:
            material_menu.addnew(name, callback, matcode=matcode )
        pass
        material_menu.update()  

        
    def update_flags_menu(self):
        """
        Populates flags menu, defining callbacks 
        """
        log.debug("update_flags_menu")

        flags_menu = self.rootmenu.find_submenu("flags")
        assert flags_menu == self.flags_menu

        flags_menu.addnew("ANY", self.flags_callback )
        for name in sorted(PHOTON_FLAGS, key=lambda _:PHOTON_FLAGS[_]):
            log.debug("update_flags_menu %s " % name )
            flags_menu.addnew(name, self.flags_callback )
        pass
        flags_menu.update()  

    def update_history_menu(self, history  ):
        """
        :param history:
        """
        self._update_history_menu( history )

    def update_history_menu_old(self, photons  ):
        """
        :param photons: chroma.event.Photons instance
        """
        nflag, history = photons.history() 
        self._update_history_menu( history )
        
    def _update_history_menu(self, history ):
        """
        :param history: array of arrays containing all unique masks and corresponding counts  

            array([[   1,   15],
                   [   2, 1282],
                   [  18,   49],
                   [  33,    1],
                   ...

        """
        history_menu = self.rootmenu.find_submenu("history")
        assert history_menu == self.history_menu

        nflag = history[:,1].sum()
        #log.info("_update_history_menu : nflag %s unique flag combinations len(history) %s " % (nflag, len(history)))

        history_menu.addnew( "ANY", self.history_callback, mask=None )
        for mask,count in sorted(history,key=lambda _:_[1], reverse=True):
            frac = float(count)/nflag
            title = "[0x%x] %d (%5.2f): %s " % (mask, count, frac, mask2arg_(mask)) 
            history_menu.addnew( title, self.history_callback, mask=mask )
        pass
        history_menu.update()
 



    def history_callback(self, item):
        """
        Choosing a "photons/history" menu item targets this action resulting 
        in selection of all photons the corresponding history bits.
        """
        self.param.mask = None
        self.param.bits = item.extra['mask']  
        self.rootmenu.dispatch('on_needs_redraw')

    def flags_callback(self, item ):
        """
        Choosing a "photons/flags" menu item targets this action resulting 
        in selectin of all photons with this flag set.
        """
        name = item.title
        allowed = PHOTON_FLAGS.keys() + ['ANY']
        assert name in allowed, name
        log.info("flags_callback setting config.args.mask to %s " % name )
        if name == 'ANY':name = 'NONE'
        self.param.mask = name 
        self.param.bits = None 
        self.rootmenu.dispatch('on_needs_redraw')


if __name__ == '__main__':
    pass


