PyOpenGL GLUT Menu
===================


Menuing Options
----------------

#. GLUT (triggering not easy)
#. python menu ?
#. add dependancy GUI toolkit 



glutAttachMenu
----------------

* :google:`glutAttachMenu`

  * https://www.opengl.org/documentation/specs/glut/spec3/node44.html  

Usage::

    void glutAttachMenu(int button);
    void glutDetachMenu(int button);


glutAttachMenu attaches a mouse button for the current window to the identifier
of the current menu; glutDetachMenu detaches an attached mouse button from the
current window. By attaching a menu identifier to a button, the named menu will
be popped up when the user presses the specified button. button should be one
of GLUT_LEFT_BUTTON, GLUT_MIDDLE_BUTTON, and GLUT_RIGHT_BUTTON. Note that the
menu is attached to the button by identifier, not by reference.


OSX Trackpad and GLUT buttons
------------------------------

=================  ==============================
GLUT_XXX_BUTTON     trackpad menu triggering 
=================  ==============================
 LEFT               light tap  
 MIDDLE             *unable to trigger it*  
 RIGHT              two finger press 
=================  ==============================


* http://www.opengl.org/resources/libraries/glut/spec3/node50.html


OSX GLUT
---------

* http://iihm.imag.fr/blanch/software/glut-macosx/


Examples
----------

::

    delta:e blyth$ pyopengl-
    delta:e blyth$ pyopengl-demo-cd

    delta:PyOpenGL-Demo blyth$ find . -name '*.py' -exec grep -l Menu {} \;
    ./GLE/maintest.py
    ./GLUT/glutplane.py
    ./proesch/simple/simpleInteraction.py
    ./tom/demo.py


GLE/maintest.py
-----------------

2-finger click on trackpad with cursor inside the GLUT window raises the EXIT button menu (this is **GLUT_RIGHT_BUTTON**)::

    (chroma_env)delta:~ blyth$ pyopengl-
    (chroma_env)delta:~ blyth$ python $(pyopengl-demo-dir)/GLE/texas.py

`GLE/maintest.py`::

    def JoinStyle (msg):
        sys.exit(0)
       
    def main(DrawStuff):
        global glutDisplayFunc, glutMotionFunc
        # initialize glut 
        glutInit(sys.argv)
        glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutCreateWindow("basic demo")
        glutDisplayFunc(DrawStuff)
        glutMotionFunc(MouseMotion)

        # create popup menu */
        glutCreateMenu (JoinStyle)
        glutAddMenuEntry ("Exit", 99) 
        glutAttachMenu (GLUT_RIGHT_BUTTON)

        # initialize GL */
        glClearDepth (1.0)
        glEnable (GL_DEPTH_TEST)
        glClearColor (0.0, 0.0, 0.0, 0.0)
        glShadeModel (GL_SMOOTH)

        glMatrixMode (GL_PROJECTION)


GLUT/glutplane.py
------------------


Hmm, a 4 item menu : but its not very easy to use (two finger clicking 
and menu selection would be tricky with a large menu).

::

    (chroma_env)delta:GLUT blyth$ python $(pyopengl-demo-dir)/GLUT/glutplane.py
    RIGHT-CLICK to display the menu.


`GLUT/glutplane.py`::


    VOID, ADD_PLANE, REMOVE_PLANE, MOTION_ON, MOTION_OFF, QUIT = range(6)

    def domotion_on(): 
        moving = GL_TRUE
        glutChangeToMenuEntry(3, "Motion off", MOTION_OFF)
        glutIdleFunc(animate)
        return

    def domotion_off():
        moving = GL_FALSE
        glutChangeToMenuEntry(3, "Motion", MOTION_ON)
        glutIdleFunc(None)
        return

    def doquit():
        sys.exit(0)
        return

    menudict ={ADD_PLANE : add_plane,
               REMOVE_PLANE : remove_plane,
               MOTION_ON : domotion_on,
               MOTION_OFF: domotion_off,
               QUIT : doquit}

    def dmenu(item):
        menudict[item]()
        return 0

    ///
    ///  dmenu is the menu callback that gets called with the menu entry enum integer
    ///

    glutCreateMenu(dmenu)
    glutAddMenuEntry("Add plane", ADD_PLANE)
    glutAddMenuEntry("Remove plane", REMOVE_PLANE)
    glutAddMenuEntry("Motion", MOTION_ON)
    glutAddMenuEntry("Quit", QUIT)
    glutAttachMenu(GLUT_RIGHT_BUTTON)

    # setup OpenGL state 



`$(pyopengl-demo-dir)/proesch/simple/simpleInteraction.py`::

    097 MENU_RED, MENU_GREEN, MENU_BLUE, MENU_QUIT = 0, 1, 2, 3
    098 colors = [ [1, 0, 0], [0, 1, 0], [0, 0, 1] ]
    099 def handleMenu( selection ):
    100     """Callback function (menu).
    101 
    102     Glut menus are not supported in PyOpenGL-3.0.0a6"""
    103 
    104     global colors, rP
    105     if selection == MENU_QUIT:
    106         sys.exit()
    107     elif MENU_RED <= selection <= MENU_BLUE:
    108         rP.drawColor = colors[ selection ]
    109     else:
    110         print 'Warning: illegel Menu entry'
    111     glutPostRedisplay( )
    112 
    113 def initMenus( ):
    114     global handleMenu
    115     colorMenu = glutCreateMenu( handleMenu )
    116     glutAddMenuEntry( "red", MENU_RED )
    117     glutAddMenuEntry( "green", MENU_GREEN )
    118     glutAddMenuEntry( "blue", MENU_BLUE )
    119     glutCreateMenu( handleMenu )
    120     glutAddSubMenu( "color", colorMenu)
    121     glutAddMenuEntry( "quit", MENU_QUIT )
    122     glutAttachMenu( GLUT_RIGHT_BUTTON )
    123 



Hmm unable to trigger the menu::

    (chroma_env)delta:pyopengl_glut blyth$ pyopengl-
    (chroma_env)delta:pyopengl_glut blyth$ python $(pyopengl-demo-dir)/proesch/simple/simpleInteraction.py




