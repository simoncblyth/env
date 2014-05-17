#!/usr/bin/env python
"""
GLUT Menus
==========

GLUT has strict stipulations on menu/submenu creation/hookup
ordering that leads to Segmentation Violations when not followed.
Pattern of glut calls needed::

    113 def initMenus( ):
    114     global handleMenu
    115     colorMenu = glutCreateMenu( handleMenu )
    116     glutAddMenuEntry( "red", MENU_RED )
    117     glutAddMenuEntry( "green", MENU_GREEN )
    118     glutAddMenuEntry( "blue", MENU_BLUE )
    119     glutCreateMenu( handleMenu )          ### TOP MENU CREATED AFTER CHILDREN
    120     glutAddSubMenu( "color", colorMenu)
    121     glutAddMenuEntry( "quit", MENU_QUIT )
    122     glutAttachMenu( GLUT_RIGHT_BUTTON )   ### ATTACHED LAST

For a menu system coming from multiple different objects 
is is not a very natural way to create a tree, the top 
menu is more naturally a fixture of the hierarchy to which 
other objects attach.

The below `DAEMenu` works around this by dividing the heirarchy creation 
from the GLUT calls and using a depth first recursive traverse to ensure child menus
are created before their parents need to "AddSubMenu" to them. This 
creation only happens when `create("LEFT/MIDDLE/RIGHT")` is called.

Refs regarding changing menus
-------------------------------

Details on changing menus

http://csclab.murraystate.edu/bob.pilgrim/515/lectures_03.html

::
  
    void glutChangeToMenuEntry(int entry, char *name, int value); 
    void glutChangeToSubMenu(int entry, char *name, int menu); 
    void glutRemoveMenuItem(int entry);  

    void glutSetMenu(int menu); 
    int glutGetMenu(void);  

    void glutMenuStatusFunc(void (*func)(int status, int x, int y); 
          status - one of GLUT_MENU_IN_USE or GLUT_MENU_NOT_IN_USE

::

    Note that we changed the menu in the keyboard callback function as opposed to
    the menu callback function. This is because we shouldn't do any changes to a
    menu while it is in use. A menu is in use until the callback is over, so we
    couldn't change the menu's structure inside the menu's own callback.

    As mentioned before, when a menu is in use it can't, or at least it shouldn't,
    be altered. In order to prevent messing up we must make sure if a menu is not
    in use before we change the menu entries. GLUT allows us to register a callback
    function that will ba called whenever a menu pops-up, and when it goes away.
    The function to register the callback is glutMenuStatusFunc.





"""
import logging, inspect
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
        self.children = []

class DAEMenu(object):
    count = 0   # class level, so all instances use unique identifiers
    def __init__(self, name, cb_argname='item' ):
        """
        :param name: menu name, only visible when a submenu
        :param cb_argname: argname of single argument callback functions or methods  
                           that require the menu item to be returned to them   
                           (could be used to have a single function/method handle
                           multiple menu items)

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
        self.cb_argname = cb_argname
        self.items = OrderedDict()
        self.menu = None
        self.children = []

    def attach_button(self, attach):
        assert attach in ('LEFT','MIDDLE','RIGHT')
        log.info("attach")
        glut.glutAttachMenu(getattr(glut,"GLUT_%s_BUTTON" % attach ))

    def create(self, attach='RIGHT'):
        self.create_MenuTree() 
        self.attach_button(attach)

    def glut_CreateMenu(self):
        log.info("glut_CreateMenu %s " % self.name)
        self.menu = glut.glutCreateMenu(self.__call__)
        for n in self.items:
            entry = self.items[n]
            log.info("glut_AddMenuEntry %s %s " % (entry.title, entry.num ))
            glut.glutAddMenuEntry( entry.title, entry.num )

    def create_MenuTree(self):
        """
        Recursively creates children(sub menus) before selves
        """
        for sub in self.children: 
            sub.create_MenuTree()
        pass
        self.glut_CreateMenu()
        pass
        for sub in self.children:
            log.info("glut_AddSubMenu %s %s " % (sub.name, sub.menu ))
            glut.glutAddSubMenu( sub.name, sub.menu ) 


    def addSubMenu(self, sub ):
        assert isinstance(sub, self.__class__ ), sub
        self.children.append(sub) 

    def add(self, title, func_or_method ):
        self.count += 1
        dmi = DAEMenuItem( self.count, title, func_or_method )
        self.items[self.count] = dmi

    def __call__(self, item ):
        if not item in self.items:
            log.warn("item %s not in DAEMenu " % item ) 
            return
        pass
        dmi = self.items[item]
        f = dmi.func_or_method 
        argspec = inspect.getargspec(f)
        args = argspec.args
        if args == ['self'] or args == []:
            f()
        elif args == ['self',self.cb_argname] or args == [self.cb_argname]:
            f(dmi)
        else:
            log.warn("cannot call menu func_or_callback as dont know what args to use %s " % repr(argspec))
            pass
        return 0     # avoids complaint from _ctypes/callbacks.c


def demo_red(item):log.info("demo_red %s " % item )
def demo_green(item):log.info("demo_green %s " % item )
def demo_blue(item):log.info("demo_blue %s " % item )


class DAEMenuDemo(object):
    def __init__(self):
        self.menu = self.make_demo_menu()

    def demo_cyan(self, item):
        log.info("demo_cyan %s " % item )

    def demo_magenta(self, item):
        log.info("demo_magenta %s " % item )

    def demo_cdo(self, item):
        log.info("demo_cdo %s " % item )

    def make_demo_menu(self):
        """
        Considerable exertions required to allow creation 
        of the menu tree without regard to GLUT.
        By splitting the glut code from the requesting code.

        Without this leads to Segmentation Violation if the 
        strictly required GLUT order is followed. 
        """
        top = DAEMenu("top")   

        primary = DAEMenu("primary")
        primary.add( "red", demo_red )
        primary.add( "green", demo_green )
        primary.add( "blue", demo_blue )

        subprime = DAEMenu("subprime")
        subprime.add("CDO", self.demo_cdo )
        primary.addSubMenu( subprime ) 

        secondary = DAEMenu("secondary")
        secondary.add( "cyan", self.demo_cyan )
        secondary.add( "magenta", self.demo_magenta )

        top.addSubMenu( primary )
        top.addSubMenu( secondary )

        return top




if __name__ == '__main__':
    pass



