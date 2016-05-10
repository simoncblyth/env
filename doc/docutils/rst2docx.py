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



from docx.oxml.shared import OxmlElement, qn



from translator import BaseTranslator

default_usage = '%prog [options] [<source> [<destination>]]'
default_description = ('Reads from <source> (default is stdin) and writes to '
                       '<destination> (default is stdout).  See '
                       '<http://docutils.sf.net/docs/user/config.html> for '
                       'the full reference.')


class AddTOC(object):
    """
    Attempt to follow the hint from 

    * https://github.com/python-openxml/python-docx/issues/36

    The document shows up, but no TOC (in Pages as least) despite the
    following gobbledegook being incorporated::

          <w:r>
            <w:fldChar w:fldCharType="begin"/>
          </w:r>
          <w:r>
            <w:instrText xml:space="preserve">TOC \* MERGEFORMAT</w:instrText>
          </w:r>
          <w:r>
            <w:fldChar w:fldCharType="end"/>
          </w:r>

    * http://stackoverflow.com/questions/9762684/how-to-generate-table-of-contents-using-openxml-sdk-2-0

    Hmm, it seems the effort required to automatically AddTOC 
    are not worth the effort. Workaround: do it manually within the client, it would 
    be necessary in any case to adjust styles of headings etc..

    """
    def __init__(self, docx):
        self.para = docx.add_paragraph()
        self.add_element(self.element('w:fldChar', {"w:fldCharType":'begin'}))
        self.add_element(self.element('w:instrText', {'xml:space':'preserve', '_text':"TOC \* MERGEFORMAT"}))
        self.add_element(self.element('w:fldChar', {'w:fldCharType':'end'}))

    def add_element(self, elem):
        run = self.para.add_run()
        r_element = run._r
        r_element.append(elem) 

    def __repr__(self):
        p_element = self.para._p
        return str(p_element.xml)

    def element(self, name, attr):
        from docx.oxml.shared import OxmlElement, qn
        elem = OxmlElement(name)  
        text = attr.pop('_text', None)
        for k, v in attr.items():
            elem.set(qn(k), v) 
        pass 
        if not text is None:
           elem.text = text 
        pass
        return elem



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
        #log.info("pformat %s " % self.document.pformat())
        #log.info("output  %s " % self.output)


    def supports(self, format):
        """This writer supports all format-specific elements."""
        return True

    def save(self, path):
        log.info("save to %s " % path)
        self.docx.save(path)

        base, ext = os.path.splitext(path)
        pxml_path = "%s%s" % (base, ".pxml")

        log.info("save pseudo-xml to %s " % pxml_path)

        with open(pxml_path, "w") as fp:
            fp.write(self.document.pformat())
        


        



def node_astext(node):
    """avoids RST source linebreaks influencing output formatting"""
    return node.astext().replace("\n", " ")  


class Text(list):
    def __init__(self, translator):
        self.translator = translator
        list.__init__(self)

    def append(self, txt, msg=None):
        list.append(self, txt)
        len = list.__len__(self)
        if self.translator.incaption:
            log.info("append %s %s %s " % (len,msg, txt))

    def pop(self, msg=None):
        txt = list.pop(self)
        len = list.__len__(self)

        if self.translator.incaption:
            for i in range(len):
                print "%4d : %s " % (i, list.__getitem__(self, i))
            pass
            log.info("pop %s %s %s " % (len,msg, txt))
        pass
        return txt
   
 

class Translator(BaseTranslator):
    def __init__(self, document, docx, writer):
        BaseTranslator.__init__(self, document)
        self.docx = docx
        self.parax = None

        self.writer = writer
        self.level = 0 

        self.intitle = False
        self.ininfo = False
        self.inraw = False
        self.incaption = False

        self.title_count = 0 
        self.caption_count = 0 
        self.pretext = None

        self.text = Text(self)
        self.para_style = []
        self.char_style = []
        self.docinfo = {}

        self.date = None
        self.newline = False 

    def astext(self):
        return ""

    def visit_section(self, node):
        self.level += 1
    def depart_section(self, node):
        self.level -= 1

    def visit_title(self, node):
        self.intitle = True
        self.title_count += 1  
    def depart_title(self, node):
        self.intitle = False


    def visit_paragraph(self, node):
        if self.ininfo:return
        self.parax = self.docx.add_paragraph("", self.parastyle)
    def depart_paragraph(self, node):
        if self.ininfo:return
        self.parax = None

    def visit_caption(self, node):
        self.caption_count += 1  
        self.incaption = True

        self.pretext = "Figure %s:" % self.caption_count
        parax = self.docx.add_paragraph("", self.parastyle)
        pfx = parax.paragraph_format 
        pfx.left_indent = Inches(0.25)
        pfx.space_after = Inches(0.25)
        log.info("caption pfx %s " % repr(dir(pfx)))
        self.parax = parax

    def depart_caption(self, node):
        self.parax = None
        self.incaption = False


    def visit_Text(self, node):
        """
        Most text appears inside paragraph, the exceptions are titles
        """
        if self.ininfo or self.inraw:return
        txt = node_astext(node)

        if self.pretext is not None:
           txt = "%s %s" % (self.pretext, txt)
           self.pretext = None 
        pass

        if self.parax is not None:
            self.parax.add_run(txt, self.charstyle)
        elif self.intitle == True:
            log.debug("visit_Text whilst intitle") 
            ## somewhat specifically add date to first title heading
            if self.title_count == 1 and not self.date is None:
                txt = "%s [%s]" % (txt, self.date)
            pass
            self.docx.add_heading(txt, self.level)
        else:
            self.report("visit_Text without parax or intitle", node) 
            self.text.append(txt, msg="visit_Text")
            assert 0
        pass


    def depart_Text(self, node):
        pass

    def visit_enumerated_list(self, node):
        # list counters are not resetting, workaround: convert source document to bulleted 
        self.para_style.append('List Number')   
    def depart_enumerated_list(self, node):
        self.para_style.pop()

    def visit_bullet_list(self, node):
        self.para_style.append('List Bullet')
    def depart_bullet_list(self, node):
        self.para_style.pop()

    def visit_raw(self, node):
        fmt = node.attributes.get('format',None)
        self.inraw = True

        txt = node_astext(node)
        if txt == "\\newline":
            #log.info("visit_raw newline spotted")
            if self.parax is not None:
                self.parax.add_run("\n", self.charstyle)
            pass
        pass
        #self.report("visit_raw", node) 
    def depart_raw(self, node):
        #self.report("depart_raw", node) 
        self.inraw = False

    def visit_literal(self, node):
        txt = node_astext(node)

        log.info("visit_literal [%s]" % txt)
        if txt == "\\n":
            log.info("literal newline spotted")

        self.report("visit_literal", node) 
        pass
    def depart_literal(self, node):
        self.report("depart_literal", node) 


    def report(self, msg, node):
        txt = node_astext(node)
        log.info("report %s %s %s " % (msg, len(self.text), txt)) 

    def _parastyle(self):
        """
        https://python-docx.readthedocs.io/en/latest/user/styles-understanding.html

        Examples of paragraph styles:
 
           List Bullet 
           List Number

        """
        if len(self.para_style): 
            style = self.para_style[-1]
        else:
            style = None
        pass
        return style
    parastyle = property(_parastyle)


    def _charstyle(self):
        """
        https://python-docx.readthedocs.io/en/latest/user/styles-understanding.html

        Examples of Character styles in default template

            Body Text Char
            Body Text 2 Char
            Body Text 3 Char
            Book Title
            Default Paragraph Font
            Emphasis
            Heading 1 Char
            Heading 2 Char
            Heading 3 Char
            Heading 4 Char
            Heading 5 Char
            Heading 6 Char
            Heading 7 Char
            Heading 8 Char
            Heading 9 Char
            Intense Emphasis
            Intense Quote Char
            Intense Reference
            Macro Text Char
            Quote Char
            Strong
            Subtitle Char
            Subtle Emphasis
            Subtle Reference
            Title Char

        """
        if len(self.char_style): 
            style = self.char_style[-1]
        else:
            style = None
        pass
        return style
    charstyle = property(_charstyle)


    def visit_emphasis(self, node):
        self.char_style.append("Emphasis")
    def depart_emphasis(self, node):
        self.char_style.pop()

    def visit_strong(self, node):
        self.char_style.append("Strong")
    def depart_strong(self, node):
        self.char_style.pop()


    def visit_figure(self, node):
        pass
    def depart_figure(self, node):
        pass

    def visit_image(self, node):
        path = self.writer.resolve(node.attributes['uri'])
        self.docx.add_picture(path)
    def depart_image(self, node):
        pass

 
    def visit_docinfo(self, node):
        self.ininfo = True
    def depart_docinfo(self, node):
        self.ininfo = False
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
 
    def visit_date(self, node):
        self.date = node_astext(node)
    def depart_date(self, node):
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

    odir = os.path.dirname(paths[1])
    if not os.path.exists(odir):
        os.makedirs(odir)




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

