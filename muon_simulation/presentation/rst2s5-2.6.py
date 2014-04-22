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

import logging, time
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import docutils.writers.s5_html as s5
from docutils.parsers.rst import directives

class MonkeyWriter(s5.Writer):
    pass

class MonkeyS5HTMLTranslator(s5.S5HTMLTranslator):
    s5_stylesheet_template = """\
<!-- configuration parameters -->
<meta name="defaultView" content="%(view_mode)s" />
<meta name="controlVis" content="%(control_visibility)s" />
<!-- style sheet links -->
<script src="%(path)s/slides.js#Q#" type="text/javascript"></script>
<link rel="stylesheet" href="%(path)s/slides.css#Q#"
      type="text/css" media="projection" id="slideProj" />
<link rel="stylesheet" href="%(path)s/outline.css#Q#"
      type="text/css" media="screen" id="outlineStyle" />
<link rel="stylesheet" href="%(path)s/print.css#Q#"
      type="text/css" media="print" id="slidePrint" />
<link rel="stylesheet" href="%(path)s/opera.css#Q#"
      type="text/css" media="projection" id="operaFix" />\n""".replace("#Q#","?monkeykillcache=%s" % time.time() )
    pass


s5.Writer = MonkeyWriter
s5.S5HTMLTranslator = MonkeyS5HTMLTranslator

from s5_video_raw import S5VideoRaw
directives.register_directive('s5_video',S5VideoRaw)
#from s5_video import S5VideoDirective
#directives.register_directive('s5_video',S5VideoDirective)


from s5_background_image import S5BackgroundImage
directives.register_directive('s5_background_image',S5BackgroundImage)




from docutils.core import publish_cmdline, default_description
description = ('Generates S5 (X)HTML slideshow documents from standalone '
               'reStructuredText sources.  ' + default_description)
publish_cmdline(writer_name='s5', description=description)


