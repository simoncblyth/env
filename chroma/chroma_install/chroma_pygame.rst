pygame
========

resources
----------

* https://bitbucket.org/pygame/pygame/issues


chroma pygame segv
--------------------

Chroma scripts are getting SEGV related to pygame, on Mavericks 10.9.1 
using macports py27-game 1.9.1_8::

    (chroma_env)delta:test blyth$ python geo.py 
    <chroma.geometry.Geometry object at 0x1092b2cd0>
    <chroma.geometry.Mesh object at 0x1092b2dd0>
    Fatal Python error: (pygame parachute) Segmentation Fault



version
--------



macports pygame
----------------

* https://trac.macports.org/browser/trunk/dports/python/py-game/Portfile

::

    In [9]: pygame.__version__
    Out[9]: '1.9.1release'


::

    delta:test blyth$ port info py27-game
    py27-game @1.9.1_8 (python, devel, multimedia, graphics)
    Variants:             portmidi, universal

    Description:          Pygame is a set of Python modules designed for writing games. It is written on top of the excellent SDL library. This allows you to create fully featured games
                          and multimedia programs in the python language. Pygame is highly portable and runs on nearly every platform and operating system.
    Homepage:             http://www.pygame.org/

    Library Dependencies: py27-numpy, libsdl_mixer, libsdl_image, libsdl_ttf
    Platforms:            darwin
    License:              LGPL-2.1+
    Maintainers:          jmr@macports.org, openmaintainer@macports.org
    delta:test blyth$ 
    delta:test blyth$ 
    delta:test blyth$ 
    delta:test blyth$ python -c "import pygame ; print pygame.__version__ "
    1.9.1release
    delta:test blyth$ 





macports trawl
------------------


* https://trac.macports.org/ticket/41930
* https://trac.macports.org/search?q=pygame
* http://www.pygame.org/docs/ref/tests.html


pygame sysfont hardcoded paths
-------------------------------


* https://bitbucket.org/pygame/pygame/issue/179/need-to-update-system-font-lists-in

  * https://bitbucket.org/pygame/pygame/commits/4ed9cb7


::

    delta:test blyth$ which fc-list
    /opt/local/bin/fc-list
    delta:test blyth$ ll /usr/X11/bin/fc-list /opt/local/bin/fc-list
    -rwxr-xr-x  1 root  wheel  48096 Nov 11 08:09 /usr/X11/bin/fc-list
    -rwxr-xr-x  1 root  admin  14000 Jan 11 13:57 /opt/local/bin/fc-list
    delta:test blyth$ 



pygame.tests font fails
--------------------------

::

    chroma_env)delta:python2.7 blyth$ python -c "import pygame ; print pygame.__file__ "
    /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pygame/__init__.pyc
    (chroma_env)delta:python2.7 blyth$ 
    (chroma_env)delta:python2.7 blyth$ 
    (chroma_env)delta:python2.7 blyth$ 
    (chroma_env)delta:python2.7 blyth$ 
    (chroma_env)delta:python2.7 blyth$ python -m pygame.tests
    skipping pygame.tests.cdrom_test (tag 'interactive')
    skipping pygame.tests.scrap_test (tag 'subprocess_ignore')
    loading pygame.tests.base_test
    loading pygame.tests.blit_test
    loading pygame.tests.bufferproxy_test
    loading pygame.tests.color_test
    loading pygame.tests.cursors_test
    loading pygame.tests.display_test
    loading pygame.tests.draw_test
    loading pygame.tests.event_test
    loading pygame.tests.fastevent_test
    loading pygame.tests.font_test
    loading pygame.tests.gfxdraw_test
    loading pygame.tests.image__save_gl_surface_test
    loading pygame.tests.image_test
    loading pygame.tests.joystick_test
    loading pygame.tests.key_test
    loading pygame.tests.mask_test
    loading pygame.tests.midi_test
    loading pygame.tests.mixer_music_test
    loading pygame.tests.mixer_test
    loading pygame.tests.mouse_test
    loading pygame.tests.movie_test
    loading pygame.tests.overlay_test
    loading pygame.tests.pixelarray_test
    loading pygame.tests.rect_test
    loading pygame.tests.sndarray_test
    loading pygame.tests.sprite_test
    loading pygame.tests.surface_test
    loading pygame.tests.surfarray_test
    loading pygame.tests.surflock_test
    loading pygame.tests.sysfont_test
    loading pygame.tests.threads_test
    loading pygame.tests.time_test
    loading pygame.tests.transform_test
    ..........................................................................................FF.............................E..............E....................................................................................................................................................................................................................................E...................................
    ======================================================================
    FAIL: FontTypeTest.test_set_bold
    ----------------------------------------------------------------------
    Traceback (most recent call last):
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pygame/tests/font_test.py", line 254, in test_set_bold
        self.failIf(f.get_bold())
    AssertionError: None

    ======================================================================
    FAIL: FontTypeTest.test_set_italic
    ----------------------------------------------------------------------
    Traceback (most recent call last):
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pygame/tests/font_test.py", line 266, in test_set_italic
        self.failIf(f.get_bold())
    AssertionError: None

    ======================================================================
    ERROR: ImageModuleTest.test_save
    ----------------------------------------------------------------------
    Traceback (most recent call last):
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pygame/tests/image_test.py", line 112, in test_save
        os.remove(temp_filename)
    OSError: [Errno 2] No such file or directory: 'tmpimg.jpg'


    ======================================================================
    ERROR: all_tests_for (pygame.tests.midi_test.AllTestCases)
    ----------------------------------------------------------------------
    Traceback (most recent call last):
      File "test/pygame.tests.midi_test.py", line 1, in all_tests_for
    subprocess completely failed with return code of 0
    cmd:          ['/usr/local/env/chroma_env/bin/python', '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pygame/tests/test_utils/test_runner.py', 'pygame.tests.midi_test', '--exclude', 'interactive,subprocess_ignore,python2_ignore', '--timings', '1']
    test_env:     {'SOURCE_TAG': 'G', 'G4NEUTRONXSDATA': '/usr/local/env/chroma_env/share/Geant4-9.5.1/data/G4NEUTRONXS1.1', 'APACHE_HOME': '/usr/local/apache/httpd-2.0.63', 'G4LEDATA': '/usr/local/env/chroma_env/share/Geant4-9.5.1/data/G4EMLOW6.23', 'G4ABLADATA': '/usr/local/env/chroma_env/share/Geant4-9.5.1/data/G4ABLA3.0', 'LC_CTYPE': 'UTF-8', 'TERM_PROGRAM_VERSION': '326', 'LIBPATH': '/usr/local/env/chroma_env/src/root-v5.34.14/lib', 'APACHE_MODE': 'source', 'LOGNAME': 'blyth', 'USER': 'blyth', 'SYSTEM_BASE': '/usr/local', 'HOME': '/Users/blyth', 'G4PIIDATA': '/usr/local/env/chroma_env/share/Geant4-9.5.1/data/G4PII1.3', 'PATH': '/Developer/NVIDIA/CUDA-5.5/bin:/usr/local/env/chroma_env/src/root-v5.34.14/bin:/usr/local/env/chroma_env/bin:/usr/local/env/chroma_env/bin:/opt/local/bin:/opt/local/sbin:/Users/blyth/env/bin:/usr/local/svn/subversion-1.4.0/bin:/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:/opt/X11/bin', 'PS1': '(chroma_env)\\h:\\W \\u\\$ ', 'LD_LIBRARY_PATH': '/usr/local/env/chroma_env/src/root-v5.34.14/lib', 'TERM_PROGRAM': 'Apple_Terminal', 'G4LEVELGAMMADATA': '/usr/local/env/chroma_env/share/Geant4-9.5.1/data/PhotonEvaporation2.2', 'SCM_FOLD': '/var/scm', 'TERM': 'xterm-256color', 'SHELL': '/bin/bash', 'VAR_BASE': '/var', 'SHLVL': '1', 'LOCAL_BASE': '/usr/local', 'VAR_BASE_BACKUP': '/var', 'DISPLAY': '/tmp/launch-NIbAPE/org.macosforge.xquartz:0', 'OUTPUT_BASE': '/tmp', 'USER_BASE': '/tmp', 'EDITOR': 'vi', 'MANPATH': '/usr/local/env/chroma_env/src/root-v5.34.14/man:/opt/local/share/man:/usr/share/man:/opt/X11/share/man:/Applications/Xcode.app/Contents/Developer/usr/share/man:/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/share/man', 'APACHE_HTDOCS': '/usr/local/apache/httpd-2.0.63/htdocs', 'SVN_EDITOR': 'vi', 'LANG': 'en_US.UTF-8', 'TERM_SESSION_ID': '7CA61AF2-CCA9-4A72-88D5-43E9CA5E19DB', 'SUDO': 'sudo', 'LOCAL_ARCH': 'Darwin', 'PYTHONPATH': '/usr/local/env/chroma_env/src/root-v5.34.14/lib', 'SSH_AUTH_SOCK': '/private/tmp/launch-G7zpn3/Listeners', 'VIRTUAL_ENV': '/usr/local/env/chroma_env', 'DYLD_LIBRARY_PATH': '/usr/local/cuda/lib:/usr/local/env/chroma_env/src/root-v5.34.14/lib:/usr/local/env/chroma_env/lib:/usr/local/svn/subversion-1.4.0/lib/svn-python/libsvn:/usr/local/svn/subversion-1.4.0/lib/svn-python/svn:', 'SVN_NAME2': 'subversion-deps-1.4.0', 'Apple_PubSub_Socket_Render': '/tmp/launch-5WqQfa/Render', 'SOURCE_NODE': 'g4pb', 'G4NEUTRONHPDATA': '/usr/local/env/chroma_env/share/Geant4-9.5.1/data/G4NDL4.0', 'LC_ALL': 'en_US.UTF-8', 'G4REALSURFACEDATA': '/usr/local/env/chroma_env/share/Geant4-9.5.1/data/RealSurface1.0', '_': '/usr/local/env/chroma_env/bin/python', 'NODE_TAG': 'D', 'TMPDIR': '/var/folders/qm/1p5gh0x94l3b0xqc8dpr9yn40000gn/T/', 'LOCAL_NODE': 'delta', 'OLDPWD': '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/lib-dynload', 'SVN_NAME': 'subversion-1.4.0', 'ROOTSYS': '/usr/local/env/chroma_env/src/root-v5.34.14', 'APACHE_NAME': 'httpd-2.0.63', '__CF_USER_TEXT_ENCODING': '0x1F5:0:0', 'BACKUP_TAG': 'U', 'PWD': '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7', 'PYTHON_PATH': '/usr/local/svn/subversion-1.4.0/lib/svn-python:', 'ENV_HOME': '/Users/blyth/env', 'SVN_HOME': '/usr/local/svn/subversion-1.4.0', 'G4RADIOACTIVEDATA': '/usr/local/env/chroma_env/share/Geant4-9.5.1/data/RadioactiveDecay3.4', 'SHLIB_PATH': '/usr/local/env/chroma_env/src/root-v5.34.14/lib', '__CHECKFIX1436934': '1', 'ENV_PREFIX': '/usr/local/env'}
    working_dir:  /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pygame
    return (top 5 lines):
    loading pygame.tests.midi_test


    ======================================================================
    ERROR: SurfarrayModuleTest.test_make_surface
    ----------------------------------------------------------------------
    Traceback (most recent call last):
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pygame/tests/surfarray_test.py", line 488, in test_make_surface
        surf = pygame.surfarray.make_surface(self._make_src_array3d(dtype))
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pygame/surfarray.py", line 243, in make_surface
        return numpysf.make_surface (array)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pygame/_numpysurfarray.py", line 368, in make_surface
        blit_array (surface, array)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pygame/_numpysurfarray.py", line 437, in blit_array
        surface.get_buffer ().write (data, 0)
    IndexError: bytes to write exceed buffer size

    ----------------------------------------------------------------------
    Ran 401 tests in 16.231s

    FAILED (failures=2, errors=3)

    (chroma_env)delta:python2.7 blyth$ 



failing font tests
~~~~~~~~~~~~~~~~~~~~~~

* http://www.pygame.org/docs/ref/font.html

Font problems are suggestive. There were install problems with freetype headers and the root install 
that forced a kludged freetype symbolic link.

Both these are failing, as cannot unbold::

    252     def test_set_bold(self):
    253         f = pygame.font.Font(None, 20)
    254         self.failIf(f.get_bold())
    255         f.set_bold(True)
    256         self.failUnless(f.get_bold())
    257         f.set_bold(False)
    258         self.failIf(f.get_bold())
    259 
    260     def test_set_italic(self):
    261         f = pygame.font.Font(None, 20)
    262         self.failIf(f.get_italic())
    263         f.set_italic(True)
    264         self.failUnless(f.get_italic())
    265         f.set_italic(False)
    266         self.failIf(f.get_bold())


Default font is bold and cannot be unbolded ?::

    In [1]: import pygame

    In [3]: pygame.font.init()

    In [4]: f = pygame.font.Font(None, 20)

    In [5]: f.get_bold()
    Out[5]: 1

    In [7]: f.set_bold(True)

    In [8]: f.get_bold()
    Out[8]: 1

    In [9]: f.set_bold(False)

    In [10]: f.get_bold()
    Out[10]: 1



Perhaps default font is not located by python/pygame ?::

    In [22]: pygame.font.get_default_font()
    Out[22]: 'freesansbold.ttf'

::

    chroma_env)delta:python2.7 blyth$ mdfind freesansbold.ttf
    /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pygame/freesansbold.ttf
    /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pygame/tests/font_test.py
    /opt/local/share/py27-matplotlib/examples/api/font_file.py
    /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/py2app/recipes/pygame.py


Use of virtual env may be exacerbating problems




freesansbold.ttf
~~~~~~~~~~~~~~~~~~

Opening the font file brings up `Font Book` with sample `The quick brown fox jumps over the lazy dog`
which says that the font is not installed, and 
provides a button to install it.


::

    open /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pygame/freesansbold.ttf

    delta:install blyth$ ll /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pygame/*.ttf
    -rw-r--r--  1 root  wheel  98600 Aug 15  2002 /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pygame/freesansbold.ttf
    delta:install blyth$ 


Why the lonely font inside pygame module ? 

* https://trac.macports.org/ticket/17749

* :google:`VIRTUALENV pygame freesansbold.ttf`


From Font Book help, fonts get installed into `~/Library/Fonts/`.

  * where is XQuartz looking for fonts ?





::

    In [3]: pygame.font.get_fonts?
    Type:       function
    String Form:<function get_fonts at 0x10d5ba398>
    File:       /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pygame/sysfont.py
    Definition: pygame.font.get_fonts()
    Docstring:
    pygame.font.get_fonts() -> list
    get a list of system font names

    Returns the list of all found system fonts. Note that
    the names of the fonts will be all lowercase with spaces
    removed. This is how pygame internally stores the font
    names for matching.

    In [4]: pygame.font.get_fonts()
    Out[4]: 
    [u'devanagarisangammn',
     u'stsong',
     u'noteworthy',
     u'headlinea',
     u'laomn',
     u'heititc',
     u'marion',
     u'luxiserif',
     u'thonburi',



Huh, default font not in the list::

    In [5]: sorted(pygame.font.get_fonts())
    Out[5]: 
    [u'albayan',
     u'alnile',
     ...
     u'didot',
     u'dinalternate',
     u'dincondensed',
     u'diwankufi',
     u'diwanthuluth',
     u'euphemiaucas',
     u'farah',
     u'farisi',
     u'futura',
     u'gb18030bitmap',
     u'geezapro',
     u'georgia',
     u'gillsans',
     u'gohatibebzemen',
     u'gujaratimt',



Where pygame finds its fonts::

    In [13]: print "\n".join(map(lambda f:"%-25s %s " % (f,pygame.font.match_font(f)),sorted(pygame.font.get_fonts())))
    albayan                   /Library/Fonts/AlBayan.ttf 
    alnile                    /Library/Fonts/Al Nile.ttc 
    altarikh                  /Library/Fonts/Al Tarikh.ttc 
    americantypewriter        /Library/Fonts/AmericanTypewriter.ttc 
    andalemono                /Library/Fonts/Andale Mono.ttf 
    apple                     /System/Library/Fonts/Apple Color Emoji.ttf 
    applebraille              /System/Library/Fonts/Apple Braille Outline 6 Dot.ttf 
    applechancery             /Library/Fonts/Apple Chancery.ttf 
    applegothic               /Library/Fonts/AppleGothic.ttf 
    appleligothic             /Library/Fonts/Apple LiGothic Medium.ttf 
    applelisung               /Library/Fonts/Apple LiSung Light.ttf 
    applemyungjo              /Library/Fonts/AppleMyungjo.ttf 
    applesymbols              /System/Library/Fonts/Apple Symbols.ttf 
    aquakana                  /System/Library/Fonts/AquaKana.ttc 
    arial                     /Library/Fonts/Arial.ttf 
    arialblack                /Library/Fonts/Arial Black.ttf 
    arialhebrew               /Library/Fonts/ArialHB.ttc 
    arialnarrow               /Library/Fonts/Arial Narrow.ttf 
    arialroundedmtbold        /Library/Fonts/Arial Rounded Bold.ttf 
    arialunicodems            /Library/Fonts/Arial Unicode.ttf 
    athelas                   /Library/Fonts/Athelas.ttc 
    avenir                    /System/Library/Fonts/Avenir.ttc 
    avenirnext                /System/Library/Fonts/Avenir Next.ttc 
    avenirnextcondensed       /System/Library/Fonts/Avenir Next Condensed.ttc 
    ayuthaya                  /Library/Fonts/Ayuthaya.ttf 
    baghdad                   /Library/Fonts/Baghdad.ttf 
    banglamn                  /Library/Fonts/Bangla MN.ttc 
    banglasangammn            /Library/Fonts/Bangla Sangam MN.ttc 
    baolisc                   /Library/Fonts/Baoli.ttc 
    baskerville               /Library/Fonts/Baskerville.ttc 
    beirut                    /Library/Fonts/Beirut.ttc 
    biaukai                   /Library/Fonts/BiauKai.ttf 
    bigcaslon                 /Library/Fonts/BigCaslon.ttf 
    bitstreamverasans         /usr/X11/lib/X11/fonts/TTF/Vera.ttf 
    bitstreamverasansmono     /usr/X11/lib/X11/fonts/TTF/VeraMono.ttf 
    bitstreamveraserif        /usr/X11/lib/X11/fonts/TTF/VeraSe.ttf 
    brushscriptmt             /Library/Fonts/Brush Script.ttf 
    chalkboard                /Library/Fonts/Chalkboard.ttc 
    chalkboardse              /Library/Fonts/ChalkboardSE.ttc 
    chalkduster               /Library/Fonts/Chalkduster.ttf 
    charter                   /Library/Fonts/Charter.ttc 
    cochin                    /Library/Fonts/Cochin.ttc 
    comicsansms               /Library/Fonts/Comic Sans MS.ttf 
    copperplate               /Library/Fonts/Copperplate.ttc 
    ...


No default font::

    In [15]: pygame.font.match_font('freesansbold') == None
    Out[15]: True

* http://echochamber.me/viewtopic.php?f=11&t=26064




pygame built upon SDL
------------------------

::

    delta:test blyth$ sdl-config --help
    Usage: sdl-config [--prefix[=DIR]] [--exec-prefix[=DIR]] [--version] [--cflags] [--libs] [--static-libs]
    delta:test blyth$ 
    delta:test blyth$ sdl-config --cflags
    -I/opt/local/include/SDL -D_GNU_SOURCE=1 -D_THREAD_SAFE
    delta:test blyth$ 
    delta:test blyth$ sdl-config --libs
    -L/opt/local/lib -lSDLmain -Wl,-framework,AppKit -lSDL -Wl,-framework,Cocoa
    delta:test blyth$ 
    delta:test blyth$ sdl-config --prefix
    /opt/local
    delta:test blyth$ sdl-config --exec-prefix
    /opt/local
    delta:test blyth$ sdl-config --version
    1.2.15
    delta:test blyth$ sdl-config --static-libs
    -L/opt/local/lib /opt/local/lib/libSDLmain.a -Wl,-framework,AppKit /opt/local/lib/libSDL.a -L/opt/local/lib -lX11 -lXext -lXrandr -lXrender -Wl,-framework,OpenGL -Wl,-framework,Cocoa -Wl,-framework,ApplicationServices -Wl,-framework,Carbon -Wl,-framework,AudioToolbox -Wl,-framework,AudioUnit -Wl,-framework,IOKit



pygame sysfont access
-----------------------


listing fonts fc-list
~~~~~~~~~~~~~~~~~~~~~~

::

    delta:test blyth$ /usr/X11/bin/fc-list | grep freesansbold
    delta:test blyth$ /usr/X11/bin/fc-list | grep ttf
    /opt/X11/share/fonts/TTF/VeraMoBI.ttf: Bitstream Vera Sans Mono:style=Bold Oblique
    /usr/X11R6/lib/X11/fonts/TTF/luxirr.ttf: Luxi Serif:style=Regular
    /Library/Fonts/MshtakanBold.ttf: Mshtakan:style=Bold,粗體,Fed,Fett,Puolilihava,Gras,Grassetto,ボールド,볼드체,Vet,Fet,Negrito,Жирный,粗体,Negrita
    /Library/Fonts/华文仿宋.ttf: STFangsong:style=Regular,標準體,Ordinær,Normal,Normaali,Regolare,レギュラー,일반체,Regulier,Обычный,常规体
    /Library/Fonts/Apple LiSung Light.ttf: Apple LiSung:style=Light



Perusing pygame.sysfont looks like combines default locations and paths returned by fc-list::

    In [6]: pygame.sysfont?
    Type:       module
    String Form:<module 'pygame.sysfont' from '/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pygame/sysfont.pyc'>
    File:       /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pygame/sysfont.py
    Docstring:  sysfont, used in the font module to find system fonts

    In [7]: pygame.sysfont??




font book installation
------------------------

Tried installing freesansbold.ttf with Font Book, after setting prefs to install at system (not user) level.
A validation warning occurred::

   'kern' table structure and contents


After installation::

    delta:install blyth$ mdfind freesansbold.ttf | grep -v Safari
    /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pygame/freesansbold.ttf
    /Library/Fonts/freesansbold.ttf
    /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/pygame/tests/font_test.py
    /opt/local/share/py27-matplotlib/examples/api/font_file.py
    /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/py2app/recipes/pygame.py
    delta:install blyth$ 


Show up as freesans::

    In [7]: print "\n".join(map(lambda f:"%-25s %s " % (f,pygame.font.match_font(f)),sorted(filter(lambda n:n[0]=='f',pygame.font.get_fonts()))))
    farah                     /Library/Fonts/Farah.ttc 
    farisi                    /Library/Fonts/Farisi.ttc 
    freesans                  /Library/Fonts/freesansbold.ttf 
    futura                    /Library/Fonts/Futura.ttc 

    In [8]: pygame.font.match_font("freesans")
    Out[8]: u'/Library/Fonts/freesansbold.ttf'






