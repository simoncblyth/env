can-presentation-machinery-be-made-to-operate-without-macOS-laptop
=====================================================================


Background 
---------------

* docutils rst2s5 
  
  * https://docutils.sourceforge.io/docs/user/slide-shows.html


* S5 xhtml slides

  * https://meyerweb.com/eric/tools/s5/


Customizations
----------------

Over the years, I customized this at most every level

* tied to py2.7 (probably due to breakages it didnt want to fix)
* custom python to transform the RST source into S5 XHTML format 
* custom javascripts + stylesheets to change features of the html render in Safari, eg page numbers, navigation features 
* custom bash to take PNG screenshots of the Safari rendered html 
* custom applescript to combine the PNG into PDF 

Unclear whether worth the effort to get this working on Linux 
while laptop still struggling along.  


Issue 1 : need to be able to get rst2s5 (from docutils ?) to operate with newer python rst2s5-2.6.py 
------------------------------------------------------------------------------------------------------

::

    TXT opticks_20240606_ihep_panel_30min.txt
    OTXT /home/blyth/simoncblyth.bitbucket.io/env/presentation/opticks_20240606_ihep_panel_30min.txt
    ODEF /home/blyth/simoncblyth.bitbucket.io/env/presentation/my_s5defs.txt
    OHTML /home/blyth/simoncblyth.bitbucket.io/env/presentation/opticks_20240606_ihep_panel_30min.html --visible-controls
    mkdir -p /home/blyth/simoncblyth.bitbucket.io/env/presentation && DOCBASE=/home/blyth/simoncblyth.bitbucket.io /opt/local/bin/python2.7 ./rst2s5-2.6.py --traceback --footnote-references="brackets" --theme-url ui/my-small-white --current-slide --language=en /home/blyth/simoncblyth.bitbucket.io/env/presentation/opticks_20240606_ihep_panel_30min.txt /home/blyth/simoncblyth.bitbucket.io/env/presentation/opticks_20240606_ihep_panel_30min.html
    /bin/sh: /opt/local/bin/python2.7: No such file or directory
    make: *** [/home/blyth/simoncblyth.bitbucket.io/env/presentation/opticks_20240606_ihep_panel_30min.html] Error 127
    gio: Unknown option -a

    Usage:
      gio open LOCATION…



YEP, ~/env/presentation/rst2s5-2.6.py::

     01 #!/usr/bin/env python
      2 # -*- coding: utf-8 -*-
      3 """
      4 Adapted from the docutils rst2s5.py tool 
      5 
      6 A minimal front end to the Docutils Publisher, producing HTML slides using
      7 the S5 template system.
      8 
      9 * http://docutils.sourceforge.net/docs/howto/rst-roles.html
     10 
     11 TODO:
     12 
     13 #. vanilla RST extlinks, similar to sphinx.ext.extlinks
     14    /opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/sphinx/ext/extlinks.py
     15 
     16 
     17 """
     18 
     19 from __future__ import print_function
     20 try:
     21     import locale
     22     locale.setlocale(locale.LC_ALL, '')
     23 except:
     24     pass
     25 
     26 try:
     27     import IPython as IP
     28 except:
     29     IP = None
     30 
     31 
     32 
     33 import os, sys, logging, codecs, re, textwrap
     34 log = logging.getLogger(__name__)
     35 
     36 
     37 #FMT  = '[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
     38 FMT = '{%(pathname)s:%(lineno)d} %(levelname)s - %(message)s'
     39 logging.basicConfig(level=logging.INFO, format=FMT)
     40 
     41 import docutils.nodes as nodes
     42 from docutils.core import publish_doctree
     43 
     ..


    


