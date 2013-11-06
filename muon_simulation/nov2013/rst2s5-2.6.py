#!/opt/local/Library/Frameworks/Python.framework/Versions/2.6/Resources/Python.app/Contents/MacOS/Python

# $Id: rst2s5.py 4564 2006-05-21 20:44:42Z wiemann $
# Author: Chris Liechti <cliechti@gmx.net>
# Copyright: This module has been placed in the public domain.

"""
A minimal front end to the Docutils Publisher, producing HTML slides using
the S5 template system.

* http://docutils.sourceforge.net/docs/howto/rst-roles.html

"""

try:
    import locale
    locale.setlocale(locale.LC_ALL, '')
except:
    pass



from docutils.core import publish_cmdline, default_description

##################
### pull out the raw role to try to modify it 
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
    if not inliner.document.settings.raw_enabled:
        msg = inliner.reporter.warning('raw (and derived) roles disabled')
        prb = inliner.problematic(rawtext, rawtext, msg)
        return [prb], [msg]
    roles.set_classes(options)

    rl = RawLink(url=text,style="font-size:12pt;font-style:monaco;")
    log.info("rawtext : %s " % rawtext )
    log.info("   text : %s " %  text )
    uu = utils.unescape(str(rl), 1)
    log.info("     uu: %s " %  uu )
    node = nodes.raw(rawtext, uu, **options)
    log.info("     node: %s " %  node )
    node.source, node.line = inliner.reporter.get_source_and_line(lineno)
    return [node], []
pass

rawlink_role.options = {'format': directives.unchanged}
roles.register_local_role('rawlink', rawlink_role)
##################




description = ('Generates S5 (X)HTML slideshow documents from standalone '
               'reStructuredText sources.  ' + default_description)

publish_cmdline(writer_name='s5', description=description)



