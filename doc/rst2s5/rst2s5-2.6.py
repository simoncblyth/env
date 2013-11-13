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


##################   pull out the raw role to try to modify it 
import logging
log = logging.getLogger(__name__)
from docutils import nodes, utils
from docutils.parsers.rst import roles, directives
logging.basicConfig(level=logging.INFO)

class RawLink(dict):
    tmpl = r"""<a href="%(url)s" style="%(style)s" > %(url)s</a>"""
    def __init__(self, *args, **kwa):
        dict.__init__(self, *args, **kwa)
    __str__ = lambda self:self.tmpl % self

def rawlink_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    """
    TODO: adapt fontsize to length of the text (ie the url)
    """
    if not inliner.document.settings.raw_enabled:
        msg = inliner.reporter.warning('raw (and derived) roles disabled')
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]
    roles.set_classes(options)

    rl = RawLink(url=text,style="font-size:15pt;font-style:monaco;")
    log.debug("rawtext : %s " % rawtext )
    log.debug("   text : %s " %  text )
    uu = utils.unescape(str(rl), 1)
    log.debug("     uu: %s " %  uu )
    node = nodes.raw(rawtext, uu, **options)
    log.debug("     node: %s " %  node )
    node.source, node.line = inliner.reporter.get_source_and_line(lineno)
    return [node], []
pass

rawlink_role.options = {'format': directives.unchanged}
roles.register_local_role('rawlink', rawlink_role)
##################




import docutils.writers.s5_html as s5




class MonkeyWriter(s5.Writer):
    pass

import time
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


from docutils.core import publish_cmdline, default_description
description = ('Generates S5 (X)HTML slideshow documents from standalone '
               'reStructuredText sources.  ' + default_description)
publish_cmdline(writer_name='s5', description=description)


