#!/usr/bin/env python

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
        Having menus coming and going is problematic, so create tree of placeholder submenus
        """ 
        self.rootmenu = rootmenu
        self.param = param
        self.setup_menus()

    def setup_menus(self):
        """
        Just structure, not content
        """
        log.info("setup_menus")
        photons_menu = DAEMenu("photons")
        flags_menu = DAEMenu("flags")
        history_menu = DAEMenu("history")
        photons_menu.addSubMenu(flags_menu)
        photons_menu.addSubMenu(history_menu) 
        self.rootmenu.addSubMenu(photons_menu)

        self.photons_menu = photons_menu
        self.flags_menu = flags_menu
        self.history_menu = history_menu

    def update(self, photons):
        log.info("update")
        self.update_flags_menu()    
        self.update_history_menu( photons )    

    def update_flags_menu(self):
        """
        """
        log.info("update_flags_menu")
        flags_menu = self.rootmenu.find_submenu("flags")
        assert flags_menu == self.flags_menu

        flags_menu.addnew("ANY", self.flags_callback )
        for name in sorted(PHOTON_FLAGS, key=lambda _:PHOTON_FLAGS[_]):
            log.info("update_flags_menu %s " % name )
            flags_menu.addnew(name, self.flags_callback )
        pass
        flags_menu.update()

    def update_history_menu(self, photons  ):
        history_menu = self.rootmenu.find_submenu("history")
        assert history_menu == self.history_menu

        nflag, history = photons.history() 
        log.info("update_history_menu : nflag %s unique flag combinations len(history) %s " % (nflag, len(history)))

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


