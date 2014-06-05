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


glutMenuStatusFunc
-------------------

* https://www.opengl.org/resources/libraries/glut/spec3/node62.html
* http://pyopengl.sourceforge.net/documentation/manual-3.0/glutMenuStatusFunc.html

Glumpy sets other glut callbacks in /usr/local/env/graphics/glumpy/glumpy/glumpy/window/backend_glut.py 




Menus API
----------

* http://openglut.sourceforge.net/group__menus.html




"""
import logging, inspect
from glumpy.window import event
import OpenGL.GLUT as glut

log = logging.getLogger(__name__)

try:
    from collections import OrderedDict
except ImportError:
    OrderedDict = dict


class DAEMenuItem(object):
    def __init__(self, num, title, func_or_method, **extra):
        self.num = num
        self.title = title
        self.func_or_method = func_or_method
        self.children = []
        self.ipos = None   # position down the items within each menu
        self.extra = extra

    def __repr__(self):
        return "DMI %s %s %s %s " % ( self.num, self.title, self.ipos, repr(self.extra) )


class DAEMenu(event.EventDispatcher):
    """
    DAEMenu lifecycle controlled from main.  

    #. instantianted very early and tacked onto the config for wide access.
    #. actual glut menu is created late, after DAEScene instantiation by the `create()` method, 
       allowing any "DAEScene" component object to add submenus to the tree

    Submenus currently created at instantiation of `DAEEvent` and `DAEPhotons` 
    and hooked up to top menu with something like::

       self.config.rmenu.addSubMenu(self.make_submenu()) # RIGHT menu hookup

    Submenus are changed (eg the history submenu) by multiple "addnew"
    (which issue no glut calls) followed by `replace_menu_items()`

        self.history.addnew( "ANY", self.history_callback, mask=None )
        ...
        self.history.replace_menu_items()


    Handles:

    #. non-glut representation of menu tree, assigning item unique identifiers
    #. steering callbacks in __call__
    #. holding future menu entries

    NB no use of GLUT allowed in here 
    """
    count = 0   # class level, so all instances use unique identifiers
    def __init__(self, name, cb_argname='item', backend=None ):
        """
        :param name: menu name, only visible when a submenu
        :param cb_argname: argname of single argument callback functions or methods  
                           that require the menu item to be returned to them   
                           (could be used to have a single function/method handle
                           multiple menu items)

        :param backend: instance of the paired underlying menu system, currently only `DAEMenuGLUT`

        """
        self.name = name
        self.cb_argname = cb_argname
        self.items = OrderedDict()
        self.newitems = OrderedDict()
        self.children = []
        self.parent = None
        self.menu = None   # index of menu, used by backends like glut, just an index so allowable
        self.backend = backend 

    def find_submenu(self, name):
        return self.find_submenu_fn( lambda _:_.name == name )

    def find_submenu_byindex(self, menu):
        return self.find_submenu_fn( lambda _:_.menu == menu )

    def find_submenu_fn(self, select_ ):
        log.info("find_submenu_fn starting at %s" % repr(self))
        result = []
        self._find_submenu( select_, result )
        assert len(result) == 1, ( len(result), result )
        return result[0] 

    def _find_submenu(self, select_, result=[]):
        if select_(self):
            result.append(self)
        for child in self.children:
            child._find_submenu(select_, result) 

    def update(self):
        """
        Dont like passthoughs like this in general, but its convenient in this case
        """
        self.top.backend.update( self ) 

    def add(self, title, func_or_method, **extra):
        self.count += 1
        dmi = DAEMenuItem( self.count, title, func_or_method, **extra )
        self.items[self.count] = dmi

    def addnew(self, title, func_or_method, **extra):
        self.count += 1
        dmi = DAEMenuItem( self.count, title, func_or_method, **extra )
        self.newitems[self.count] = dmi

    def addSubMenu(self, sub ):
        assert issubclass(sub.__class__,  DAEMenu ), sub    # same class gives True too
        assert sub != self, "cannot add menu to self"
        sub.parent = self
        self.children.append(sub) 
        log.info("addSubMenu %s " % repr(sub) )

    def __repr__(self):
        return "DM menu %s name %s items %s children %s " % (self.menu, self.name, len(self.items), len(self.children) )

    def dump(self, items=False):
        print self
        if items:
            for dmi in self.items:
                print dmi
        for child in self.children:
            child.dump(items=items)


    def _get_top(self):
        """
        Recursively trace up the tree to find the root DAEMenu. 
        """
        if self.parent is None:
            return self
        return self.parent._get_top()
    top = property(_get_top, doc=_get_top.__doc__)

    def __call__(self, item ):
        """
        :param item: menu item index that GLUT provides to callback 

        Method used to handle all menu callbacks.

        #. returned menu item index is used to identify the corresponding 
           DAEMenuItem to lookup the appropriate callback method or function
           
        #. introspection is used in order to prepare appropriate arguments 
           for the callback 

        """
        if not item in self.items:
            log.warn("menu item index %s not in DAEMenu " % item ) 
            return 0
        pass
        dmi = self.items[item]   # items keyed on menu item indices
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

    def dispatch(self, event_name='on_needs_redraw', event_obj=None):
        log.info("dispatch %s %s " % (event_name, event_obj))
        self.dispatch_event(event_name, event_obj)
    pass


DAEMenu.register_event_type('on_needs_redraw')



class DAEMenuGLUT(object):
    """
    Handles:

    #. creating glut menu from the non-glut representation DAEMenu
    #. menu in use callback 
    #. menu updating 

    Delaying a menu update that cannot be performed due
    to the menu being currently in used is handled via 
    the pending slot which is checked when the MenuStatus
    callback indicates that menu is no longer in use.

    In case of multiple menu updates piling up whilst the
    menu is in used, the intermediates are just skipped. 

    """
    def __init__(self):
        """
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
        self.pending = None
        self.menu_in_use = False 

    def setup_glutMenuStatusFunc(self):
        """
        Invoked from main after glumpy window/Figure creation 
        """
        glut.glutMenuStatusFunc(self.MenuStatus)

    def MenuStatus(self,status,x,y):
        """
        Callback invoked on starting/ending menu usage
        that sets the menu_in_use flag. Note that only 
        the root top menu handles this. 
        """
        if status == glut.GLUT_MENU_IN_USE:
            self.menu_in_use = True
        elif status == glut.GLUT_MENU_NOT_IN_USE:
            self.menu_in_use = False
            self.after_menu_usage()
        else:
            assert 0, status 
        pass
        log.info("MenuStatus menu_in_use %s " % self.menu_in_use)

    def after_menu_usage(self):
        log.info("after_menu_usage")
        if not self.pending is None:
            log.info("after_menu_usage proceed with pending update")
            self.update( self.pending )
        pass

    def glut_AttachMenu(self, button):
        assert button in ('LEFT','MIDDLE','RIGHT')
        log.debug("attach")
        glut.glutAttachMenu(getattr(glut,"GLUT_%s_BUTTON" % button ))

    def create(self, dmenu, button='RIGHT'):
        """
        Creats glut menu from DAEMenu instance 
        """
        dmenu.dump()

        self.create_MenuTree(dmenu) 
        self.glut_AttachMenu(button)

    def glut_CreateMenu(self, dmenu ):
        log.info("glut_CreateMenu %s " % dmenu.name)
        self.glut_AddMenuEntries(dmenu) 

    def glut_AddMenuEntries(self, dmenu):
        """
        Hmm, can writing the ipos back to the dmenu be avoided ?
        """
        #curmenu = glut.glutGetMenu()

        if dmenu.menu is None:
            log.info("glut_AddMenuEntries creating menu for %s " % repr(dmenu))
            menu = glut.glutCreateMenu(dmenu.__call__)
            dmenu.menu = menu    # glut index 
        pass

        assert dmenu.menu > 0
        self.glut_SetMenu(dmenu.menu)
        for ipos, n in enumerate(dmenu.items):
            entry = dmenu.items[n]
            entry.ipos = ipos + 1  # guessing menu position doen list  to be 1-based ? 
            log.info("glut_AddMenuEntry %s %s %s " % (entry.title, entry.num, entry.ipos))
            glut.glutAddMenuEntry( entry.title, entry.num )
        pass
        #self.glut_SetMenu( curmenu ) 

    def glut_SetMenu(self, menu):
        assert menu > 0 , menu
        log.info("glut_SetMenu %s " % menu )
        glut.glutSetMenu( menu ) 

    def glut_RemoveMenuItem(self, ipos, dmenu ):
        log.debug("glut_RemoveMenuItems %s ipos %s " % (dmenu.name, ipos))
        #curmenu = glut.glutGetMenu()
        self.glut_SetMenu(dmenu.menu)
        glut.glutRemoveMenuItem(ipos) 
        #self.glut_SetMenu( curmenu ) 

    def create_MenuTree(self, dmenu):
        """
        Recursively creates children(sub menus) before selves
        """
        for dsub in dmenu.children: 
            self.create_MenuTree(dsub)
        pass
        self.glut_CreateMenu(dmenu)
        pass
        for dsub in dmenu.children:
            log.debug("glut_AddSubMenu %s %s " % (dsub.name, dsub.menu ))
            glut.glutAddSubMenu( dsub.name, dsub.menu ) 

    def update(self, dmenu):
        """
        Formerly `replace_menu_items`

        Removes old items and replaces them with newitems
        collected with addnew

        The distinction between "items" (the live menu) 
        and their future replacement "newitems" is still
        needed as the item instances are used to identify the 
        appropriate callback. 

        """
        if self.menu_in_use:
            log.info("cannot update menu now as its in use") 
            self.pending = dmenu
            return

        log.info("update %s " % repr(dmenu))
        self.remove_menu_items(dmenu)

        while len(dmenu.newitems) > 0:
            n, entry = dmenu.newitems.popitem(last=False)
            dmenu.items[n] = entry
        pass       
        self.glut_AddMenuEntries(dmenu)

    def remove_menu_items(self, dmenu):
        """
        Removes all menu items 
        """ 
        log.info("remove_menu_items %s " % repr(dmenu))
        while len(dmenu.items)>0:
            n, entry = dmenu.items.popitem()
            ipos = entry.ipos
            if not ipos is None:
                self.glut_RemoveMenuItem(ipos, dmenu)     
            else:
                log.info("cannot remove ipos None")
            pass



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
        strictly required GLUT order is not followed. 
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

    logging.basicConfig(level=logging.INFO)

    dmd = DAEMenuDemo()
    top = dmd.menu 

    top.dump()

    m = top.find_submenu("subprime")
    print m 

    #m = top.find_submenu_byindex(2)
    #print m 




