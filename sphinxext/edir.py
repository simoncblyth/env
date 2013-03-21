#!/usr/bin/env python

import os, urllib, logging, posixpath
log = logging.getLogger(__name__)

from glob import glob
from docutils import nodes, utils
from sphinx.util import docname_join
from sphinx.util.compat import Directive

def eglobpaths( ptns , docname ):
    """
    :param ptns: space delimited list of glob patterns relative to the document position in the tree
    :param docname: doc path such as 
    :return: list of paths relative to prefixd when reldocdir is False OR relative to document dir when reldocdir is True

    """
    docdir = posixpath.normpath(posixpath.join('/'+docname,'..'))[1:]
    abs_names = []
    for ptn in ptns.split():
        patname = docname_join(docname, ptn)   # leaf trimming join, ie join the directory where the document resides with ptns relative to there
        abs_names += glob( patname )
        pass
    log.debug("\n".join(['abs_names']+abs_names))

    red_names = map(lambda _:_[len(docdir)+1:], abs_names )
    log.debug("\n".join(['red_names (relative to docdir)']+red_names))

    log.debug("globpaths %s %s => [%s] %s " % (ptns, docname, len(red_names), red_names )) 
    return red_names


def make_erefnode( docname , name ):
    """
    :param docname: 
    :param name: eref argument string
    """
    url = urllib.quote(name) 
    title = ":eref:`%s`" % name 
    pnode = nodes.reference(title, title, internal=False, refuri=url)
    return pnode


def present_list( docname, rel_names, pfx ):
    """
    :param docname:
    :param rel_names:

    #. avoid empty lists, as these break the latex render
    """
    wrap = nodes.paragraph()   
    if len(rel_names) == 0:
        txt = "edir directive yields no entries " 
        wrap += nodes.Text(txt,txt)
    else:
        wdl = nodes.bullet_list('')
        wrap += wdl
        for rname in sorted(rel_names):
            para = nodes.paragraph() 
            para += make_erefnode( docname, pfx + rname )
            item = nodes.list_item('', para )
            wdl.append(item)
        pass     
        txt = "List created by edir directive yields %s entries." % ( len(rel_names) )
        wrap += nodes.Text(txt,txt)
    return [wrap]


class EDir(Directive):
    """
    """
    has_content = True
    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec = {
       'pfx':str,
    }

    def run(self):
        """
        """
        pfx = self.options.get('pfx','')
        env = self.state.document.settings.env
        assert len(self.arguments) == 1, (self.arguments, "require space delimited glob pattern string eg '*.pdf' OR '*.pdf apple/*.pdf'  identifying a set of files to list")
        rel_names = eglobpaths( self.arguments[0], env.docname ) 
        return present_list( env.docname, rel_names, pfx )


def make_eref_role():
    def role(typ, rawtext, text, lineno, inliner, options={}, content=[]):
        """
        :param typ: eg eref
        :param rawtext: eg :eref:`hello`
        :param text: eg hello
        :param inliner:
        """
        env = inliner.document.settings.env
        pnode = make_erefnode( env.docname , text )
        return [pnode], []
    return role    

def setup_eref_roles(app):
   app.add_role('eref', make_eref_role())

def setup(app):
    global _app
    _app = app 
    app.connect('builder-inited', setup_eref_roles)
    app.add_directive('edir', EDir )

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    pfx = "http://dayabay.phys.ntu.edu.tw/tracs/heprez/trunk/qxml/indices/"
    docname = "/Users/blyth/heprez/qxml/indices/index"
    names = eglobpaths("*.xq",docname )
    p = present_list( docname , names , pfx )
    print p[0].pformat()


