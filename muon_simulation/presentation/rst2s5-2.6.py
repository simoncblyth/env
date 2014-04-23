#!/opt/local/Library/Frameworks/Python.framework/Versions/2.6/Resources/Python.app/Contents/MacOS/Python
"""
Adapted from the docutils rst2s5.py tool 

A minimal front end to the Docutils Publisher, producing HTML slides using
the S5 template system.

* http://docutils.sourceforge.net/docs/howto/rst-roles.html

"""
try:
    import locale
    locale.setlocale(locale.LC_ALL, '')
except:
    pass

import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import docutils.nodes as nodes

#from s5_html_monkey import MonkeyWriter, MonkeyS5HTMLTranslator 
#s5.Writer = MonkeyWriter
#s5.S5HTMLTranslator = MonkeyS5HTMLTranslator

def add_node(node, **kwds):
    """ 
    s5html version of sphinx/application.py:add_node 
    """
    log.debug('adding node: %r', (node, kwds))
    nodes._add_node_class_names([node.__name__])
    for key, val in kwds.iteritems():
        try:
            visit, depart = val 
        except ValueError:
            raise ExtensionError('Value for key %r must be a ' '(visit, depart) function tuple' % key)
        if key == 'html':
            from docutils.writers.s5_html import S5HTMLTranslator as translator
        else:
            continue
        setattr(translator, 'visit_'+node.__name__, visit)
        if depart:
            setattr(translator, 'depart_'+node.__name__, depart)
    pass

from docutils.parsers.rst import directives

#from s5_video_raw import S5VideoRaw
#directives.register_directive('s5_video',S5VideoRaw)

import s5_video
directives.register_directive('s5_video',s5_video.S5VideoDirective)
add_node( s5_video.s5video, 
    html=(s5_video.visit_s5video_html, s5_video.depart_s5video_html)
)

from s5_background_image import S5BackgroundImage
directives.register_directive('s5_background_image',S5BackgroundImage)

from docutils.core import publish_cmdline, default_description
description = ('Generates S5 (X)HTML slideshow documents from standalone '
               'reStructuredText sources.  ' + default_description)

publish_cmdline(writer_name='s5', description=description)


