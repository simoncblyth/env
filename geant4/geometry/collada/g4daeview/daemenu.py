#!/usr/bin/env python

import logging
import OpenGL.GLUT as glut

log = logging.getLogger(__name__)


try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict


class DAEMenuItem(object):
    def __init__(self, num, title, func_or_method ):
        self.num = num
        self.title = title
        self.func_or_method = func_or_method
    def create(self):
        glut.glutAddMenuEntry( self.title, self.num )


class DAEMenu(object):
    count = 0   # class level, so all instances use unique identifiers
    def __init__(self, name, attach=None):
        """
        :param name: menu name, only visible when a submenu
        :param attach:  None or LEFT/MIDDLE/RIGHT, use None for submenus

        To trigger GLUT_RIGHT_BUTTON on OSX Trackpad 
        requires a firm two-finger press that is not very convenient for 
        rapid menu usage.

        Adjusting preferences to "Enable Button Emulation" can configure
        key modifiers for the buttons that are a lot easier to use.
        Just need to tap trackpad whilst holding the modifier key. 
        This allows easy navigation of multi-level menus.

        To adjust preferences, while running *g4daeview*:

        #. Goto `python > Preferences... [mouse]` tab 
           and select *Enable Button Emulation* 

           The below defaults are fine:

           Right Button Modifier: control
           Middle Button Modifier: option

        """ 
        self.name = name
        self.items = OrderedDict()
        self.menu = glut.glutCreateMenu(self.__call__)

        if attach is None:
            pass
        elif attach in ('LEFT','MIDDLE','RIGHT'):
            but = getattr(glut,"GLUT_%s_BUTTON" % attach )
            glut.glutAttachMenu(but)
        else:
            assert 0
        pass

    def addSubMenu(self, sub ):
        assert isinstance(sub, self.__class__ ), sub
        glut.glutAddSubMenu( sub.name, sub.menu ) 

    def add(self, title, func_or_method ):
        self.count += 1
        dmi = DAEMenuItem( self.count, title, func_or_method )
        dmi.create()
        self.items[self.count] = dmi

    def __call__(self, item ):
        """
        https://github.com/python-git/python/blob/master/Modules/_ctypes/callbacks.c
        http://stackoverflow.com/questions/7259794/how-can-i-get-methods-to-work-as-callbacks-with-python-ctypes
        """
        if not item in self.items:
            log.warn("item %s not in DAEMenu " % item ) 
            return
        pass
        dmi = self.items[item]
        dmi.func_or_method(item)
        return 0     # avoids complaint from _ctypes/callbacks.c


if __name__ == '__main__':
    pass



