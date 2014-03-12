PyGame Examples
================

Supplied Examples
------------------


::

    (chroma_env)delta:pygame_examples blyth$ ll /opt/local/share/doc/py27-game/examples/
    total 384
    -rw-r--r--   1 root  admin   2700 Apr 26  2007 readme.txt
    -rw-r--r--   1 root  admin   2995 May 27  2009 vgrade.py
    -rw-r--r--   1 root  admin   6910 May 27  2009 testsprite.py
    -rw-r--r--   1 root  admin   2488 May 27  2009 stars.py
    -rw-r--r--   1 root  admin   1359 May 27  2009 sound.py
    -rw-r--r--   1 root  admin   6792 May 27  2009 scroll.py
    -rw-r--r--   1 root  admin   2693 May 27  2009 scrap_clipboard.py
    -rw-r--r--   1 root  admin   4438 May 27  2009 scaletest.py
    -rw-r--r--   1 root  admin   3026 May 27  2009 pixelarray.py
    -rw-r--r--   1 root  admin   1397 May 27  2009 overlay.py
    -rw-r--r--   1 root  admin   6714 May 27  2009 oldalien.py
    -rw-r--r--   1 root  admin   1509 May 27  2009 movieplayer.py
    -rw-r--r--   1 root  admin   1837 May 27  2009 moveit.py
    -rw-r--r--   1 root  admin   5563 May 27  2009 mask.py
    -rw-r--r--   1 root  admin   2481 May 27  2009 liquid.py
    -rw-r--r--   1 root  admin   1333 May 27  2009 headless_no_windows_needed.py
    -rw-r--r--   1 root  admin   2719 May 27  2009 glcube.py
    -rw-r--r--   1 root  admin   2539 May 27  2009 fonty.py
    -rw-r--r--   1 root  admin   2890 May 27  2009 fastevents.py
    -rw-r--r--   1 root  admin   3062 May 27  2009 cursors.py
    -rw-r--r--   1 root  admin   5876 May 27  2009 chimp.py
    -rw-r--r--   1 root  admin   2197 May 27  2009 camera.py
    -rw-r--r--   1 root  admin   5844 May 27  2009 blit_blends.py
    -rw-r--r--   1 root  admin   3105 May 27  2009 blend_fill.py
    -rw-r--r--   1 root  admin   4615 May 27  2009 arraydemo.py
    -rw-r--r--   1 root  admin   9554 May 27  2009 aliens.py
    -rw-r--r--   1 root  admin      0 May 27  2009 __init__.py
    -rw-r--r--   1 root  admin  29990 May 30  2009 midi.py
    -rw-r--r--   1 root  admin    762 May 31  2009 aacircle.py
    -rw-r--r--   1 root  admin   7512 Jul  7  2009 sound_array_demos.py
    -rw-r--r--   1 root  admin   3677 Jul  9  2009 eventlist.py
    drwxr-xr-x   4 root  admin    136 Jan 15 18:53 macosx
    drwxr-xr-x  33 root  admin   1122 Jan 15 18:53 data
    drwxr-xr-x  18 root  admin    612 Jan 15 18:53 ..
    drwxr-xr-x  35 root  admin   1190 Jan 15 18:53 .
    (chroma_env)delta:pygame_examples blyth$ 



rotating cube
--------------

The cube example distributed with pygame needs pyopengl. 

* :google:`pygame cube`
* http://codentronix.com/2011/05/12/rotating-3d-cube-using-python-and-pygame/

::

    curl -L http://goo.gl/5GaKx -o rotating_cube.zip
    unzip rotating_cube.zip
    cp rotating_cube.zip q_rotating_cube.py 
    perl -pi -e 's,\r,,g' q_rotating_cube.py       # fix DOS line endings




