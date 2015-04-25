#!/usr/bin/env python
"""
rst2docx.py
============

Converts a limited subset of RST source into 
OpenXML docx (ie Word XML document).

Usage::

    rst2docx.py /tmp/report.rst /tmp/report.docx


Dependencies
-------------

* python-docx see docx-


See also
---------

*docxbuilder-* which does a similar thing with Sphinx integration 
but using an obsolete version of docx and with template document
complications  

*openxml-* *mono-* based use the the C# OpenXML API, impressions
of C# : Java verbosity of code, Microsoft boring documentation  



"""

import os, logging, argparse
log = logging.getLogger(__name__)

from docx import Document
from docx.shared import Inches

from docutils import writers
from docutils.core import Publisher
import docutils.nodes as nodes

from translator import BaseTranslator

default_usage = '%prog [options] [<source> [<destination>]]'
default_description = ('Reads from <source> (default is stdin) and writes to '
                       '<destination> (default is stdout).  See '
                       '<http://docutils.sf.net/docs/user/config.html> for '
                       'the full reference.')



class Writer(writers.Writer):
    supported = ('pprint', 'pformat', 'pseudoxml')
    config_section = 'pseudoxml writer'
    config_section_dependencies = ('writers',)
    output = None

    def __init__(self, uribase):
        writers.Writer.__init__(self) 
        self.translator_class = Translator
        self.uribase = uribase
        self.docx = Document()

    def resolve(self, uri):
        path = os.path.join(self.uribase, uri)
        return os.path.abspath(path)
 
    def translate(self):
        visitor = self.translator_class(self.document, self.docx, self)
        self.document.walkabout(visitor)

        log.info("docinfo %s " % repr(visitor.docinfo))

        self.output = visitor.astext()

    def supports(self, format):
        """This writer supports all format-specific elements."""
        return True

    def save(self, path):
        log.info("save to %s " % path)
        self.docx.save(path)


def node_astext(node):
    """avoids RST source linebreaks influencing output formatting"""
    return node.astext().replace("\n", " ")  

class Translator(BaseTranslator):
    def __init__(self, document, docx, writer):
        BaseTranslator.__init__(self, document)
        self.docx = docx
        self.writer = writer
        self.level = 0 
        self.intitle = False
        self.text = []
        self.style = []
        self.docinfo = {}
        self.nopara = False

    def astext(self):
        return "klop"

    def visit_section(self, node):
        self.level += 1
    def depart_section(self, node):
        self.level -= 1

    def visit_title(self, node):
        pass
    def depart_title(self, node):
        self.docx.add_heading(self.text.pop(), self.level)

    def visit_Text(self, node):
        self.text.append(node_astext(node))
    def depart_Text(self, node):
        pass

    def visit_enumerated_list(self, node):
        # list counters are not resetting, workaround: convert source document to bulleted 
        self.style.append('List Number')   
    def depart_enumerated_list(self, node):
        self.style.pop()

    def visit_bullet_list(self, node):
        self.style.append('List Bullet')
    def depart_bullet_list(self, node):
        self.style.pop()


    def visit_paragraph(self, node):
        if self.nopara:return
        self.text.append(node_astext(node))
    def depart_paragraph(self, node):
        if self.nopara:return
        if len(self.style): 
            style = self.style[-1]
        else:
            style = None
        pass
        self.docx.add_paragraph(self.text.pop(), style)


    def visit_figure(self, node):
        pass
    def depart_figure(self, node):
        pass

    def visit_image(self, node):
        path = self.writer.resolve(node.attributes['uri'])
        self.docx.add_picture(path)
    def depart_image(self, node):
        pass

    def visit_caption(self, node):
        self.text.append(node_astext(node))
    def depart_caption(self, node):
        self.docx.add_paragraph(self.text.pop())

 
    def visit_docinfo(self, node):
        self.nopara = True
    def depart_docinfo(self, node):
        self.nopara = False
        if 'title' in self.docinfo:
            self.docx.add_heading(self.docinfo['title'], self.level)

    def visit_field(self, node):
        pass
    def depart_field(self, node):
        pass
    def visit_field_name(self, node):
        self.field_name = node_astext(node)
    def depart_field_name(self, node):
        pass
    def visit_field_body(self, node):
        if not self.field_name is None:
             field_body = node_astext(node)
             self.docinfo[self.field_name] = field_body 
             self.field_name = None
        pass 
    def depart_field_body(self, node):
        pass
 



class Config(object):
    def __init__(self, doc):
        parser = argparse.ArgumentParser(doc)
        parser.add_argument( "paths", nargs='+',  help=""  )
        self.args = parser.parse_args()



def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)-8s %(message)s" )

    uribase = os.environ.get('URIBASE',"/Library/WebServer/Documents")

    config = Config(__doc__)
    paths = config.args.paths
    assert len(paths) == 2

    log.info("reading %s " % paths[0])


    reader=None
    parser=None
    writer=Writer(uribase=uribase)

    reader_name='standalone'
    parser_name='restructuredtext'
    writer_name= None

    settings = None
    settings_spec = None
    settings_overrides = None
    config_section = None
    enable_exit_status=True

    argv=[paths[0]]
    usage=default_usage
    description=default_description
 
    pub = Publisher(reader, parser, writer, settings=settings)
    pub.set_components(reader_name, parser_name, writer_name)

    output = pub.publish(
        argv, usage, description, settings_spec, settings_overrides,
        config_section=config_section, enable_exit_status=enable_exit_status)


    writer.save(paths[1])


    return output



if __name__ == '__main__':
    main()

