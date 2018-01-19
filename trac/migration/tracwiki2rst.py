#!/usr/bin/env python
"""
"""
import logging, sys, re, os, collections, datetime, codecs, copy
log = logging.getLogger(__name__)

from env.doc.rstutil import rst2html_open    
from env.trac.migration.doclite import Para, Head, HorizontalRule, ListTagged, Toc, Literal, CodeBlock, Meta
from env.trac.migration.doclite import Anchor, Contents, Sidebar, Page, SimpleTable
        
class TracWiki2RST(object):
    """
    """ 
    skips = r"""
    [[TracNav
    """.lstrip().rstrip().split()

    @classmethod
    def meta_top(cls, wp, pg, args):

        md = wp.metadict

        if not args.vanilla:
            anchor = Anchor(wp.name, md["tags"])
            pg.append(anchor)
            sidebar = Sidebar(md)
            pg.append(sidebar) 
        pass

        contents = Contents(depth=2)
        pg.append(contents)

        meta = Meta(md)
        meta.append(":orphan:")

        if args.origtmpl is not None:
            origurl = args.origtmpl % wp.name 
            editurl = "%s?action=edit" % origurl 
            meta.append(":origurl: %s" % origurl)
            meta.append(":editurl: %s" % editurl)
        pass
        meta.append(u"")  # ensure some unicode, to kick in coercion
        pg.append(meta)

     
    @classmethod
    def dbg_tail(cls, wp, text, pg):
        """
        This would be wrong (fails with non-ascii) as it mixes byte strings and unicode::

           pg.append(Literal(str(pg).split("\n"))) 

        """
        pg0 = copy.deepcopy(pg) # avoid including these debug additions in the dump 

        pg.append(Head("%s dbg_tail" % wp.name,1))

        pg.append(Head("Literal converted rst",2))
        pg.append(CodeBlock(pg0.rst.split("\n"), lang="rst", linenos=True))

        pg.append(Head("Literal tracwiki text",2))
        pg.append(CodeBlock(text.split("\n"),lang="bash", linenos=True)) 
        # NB not wp.text as need to obey .txt file overrides of DB content, see wtracdb-edtest

        pg.append(Head("Literal repr(pg)",2))
        pg.append(CodeBlock(repr(pg0).split("\n"), lang="pycon", linenos=True))
        ## repr always fits in ascii ?

        pg.append(Head("Literal unicode(pg)",2))
        pg.append(CodeBlock(unicode(pg0).split("\n"), lang="pycon", linenos=True))

    @classmethod
    def page_from_tracwiki(cls, wp, text, args, dbg=True):
        name = wp.name
        log.info("page_from_tracwiki %s " % name)

        pg = Page(name)
        cls.meta_top(wp, pg, args) 

        conv = cls(text, pg, name=name)

        for i in pg.incomplete_instances():
            wp.complete_ListTagged(i)
        pass
        if dbg and not args.vanilla:
            cls.dbg_tail(wp, text, pg) 
        pass
        return pg 

    @classmethod
    def make_index(self, name, title, pages):
        idx = Page(name)        
        hdr = Head(title, 1)
        toc = Toc(map(lambda page:page.name,pages),maxdepth=1)

        foot = Head("indices and tables", 1)

        para = Para()
        para.append(u"")   # some unicode is needed, to get the py2 coercion to kick in 
        para.append("* :ref:`genindex`")
        para.append("* :ref:`modindex`")
        para.append("* :ref:`modindex`")
        para.append("")

        idx.append(hdr) 
        idx.append(toc) 
        idx.append(foot) 
        idx.append(para) 

        return idx

    def end_para(self):
        if self.cur_para is None:return
        self.content.append(self.cur_para)
        self.cur_para = None
    def end_literal(self):
        if self.cur_literal is None:return
        self.content.append(self.cur_literal)
        self.cur_literal = None

    def add_line(self, line):
        if self.cur_para is None:
             self.cur_para = Para()
        pass
        self.cur_para.append(line) 

    def __init__(self, text, content, name=None):
        self.orig = text
        self.content = content
        self.name = name

        lines = filter(lambda _:not(_.startswith(tuple(self.skips))), text.split("\n"))

        self.cur_para = None
        self.cur_literal = None
        self.cur_table = None

        for line in lines:
            # cur_literal None avoids looking inside literal blocks for Heads OR ListTagged
            if self.cur_literal is None and Head.is_match(line):  
                self.end_para()
                head = Head.from_line(line, name=self.name)
                self.content.append(head) 
            elif self.cur_literal is None and HorizontalRule.is_match(line):
                self.end_para()
                hr = HorizontalRule()
                self.content.append(hr) 
            elif self.cur_literal is None and ListTagged.is_match(line):
                self.end_para()
                tgls = ListTagged.from_line(line)
                self.content.append(tgls) 
            elif Literal.is_start(line):
                log.debug("start Literal")
                self.end_para()
                self.cur_literal = Literal()
            elif Literal.is_end(line):
                log.debug("end Literal")
                self.end_literal()
            elif SimpleTable.is_simpletable(line):
                # the handling of literated tables is a bit of a kludge, 
                # but to do better would need to complicate things with 
                # trees rather than the simplifying global list 
                if self.cur_table is None:
                    log.debug("start SimpleTable")
                    self.cur_table = SimpleTable(pagename=name,literal=self.cur_literal is not None) 
                    self.end_para()
                    self.end_literal()
                pass
                self.cur_table.append(line)
            elif self.cur_table is not None:  # have just left the table 
                self.content.append(self.cur_table)
                self.cur_table = None 
                self.add_line(line)   # avoid chomping the blank line following tables
            else:
                if self.cur_literal is not None:
                    self.cur_literal.append(line)
                else:
                    self.add_line(line)
                pass 
            pass
        pass
        self.end_para()


class DummyWikiPage(object):
    metadict = {'tags':"Red Green Blue", 'name':"DummyWikiPage"} 
    name = "DummyWikiPage"
 
class DummyArgs(object):
    origtmpl = None
    origurl = None
    vanilla = True

class TestSnippet(object):
    def __call__(self, txt, open_=False, skip=True):
        if skip:return
        wp = DummyWikiPage()
        args = DummyArgs()

        text = "\n".join(map(lambda _:_[4:], txt.split("\n")[1:-1]))

        pg = TracWiki2RST.page_from_tracwiki(wp, text, args)
        rst = pg.rst
        assert type(rst) is unicode

        div = "\n" +  "#" * 100 + "\n"
    
        print div
        print text
        print div
        print rst 
        print div
        print repr(pg)
        print div
        
        # hmm this only works for Vanilla RST, not Sphinx extensions
        if open_:
            rst2html_open(rst, "pg")  
        pass


if __name__ == '__main__':
    level = 'DEBUG'
    logging.basicConfig(level=getattr(logging,level))
 
    ts = TestSnippet()     

    ts(r"""
    First Line

    = Hello =

    == World ==

    * red
    * green
    * blue

    """,open_=False, skip=True)

    ts(r"""
    == Simple Table ==

    
    Text on line immediately before table causes RST indent error, but tabrst.py adds blank lines to avoid
    || red || green || blue ||
    || red || green || blue ||
    || red || green || blue ||
    || red || green || blue ||
    || red || green || blue ||

    Line after blank after table   
    """,open_=False, skip=True )


    ts(r"""
    Single column table is supported by Trac but not RST, so add blank extra column 

    || red ||
    || green ||
    || blue ||

    """,open_=False, skip=True)


    ts(r"""

    || silver   ||    ||                                                   ||          ||     ||       ||           ||
    || ..       || A  || forbidden city                                    || CF-00016 || 62  || 86.4G || raw_1018  ||
    || ..       || B  || f. city / T square / Temple of Heaven / G. wall   || CF-00017 || 124 || 84.5G || raw_1018  ||
    || blue     ||    ||                                                   ||          ||     ||       ||           ||
    || ..       || C  || G. wall / acrobats / josh and oli                 || CF-00018 || 124 || 82.6G || raw_1019  ||
    || ..       || D  ||                                                   ||          ||     ||       ||           ||
    || black    ||    ||                                                   ||          ||     ||       ||           ||
    || ..       || E  ||                                                   ||          ||     ||       ||           ||
    || ..       || F  || josh oli / xmas2007 / hkjan08                     || CF-00019 || 118 || 80.8G || raw_1019  ||

    * note the RST empty comments placeholders in 1st columns, prior to automating this table layout was messed up 
    
    """,open_=False, skip=True)

    ts(r"""

    Hmm literated tables giving: Inconsistent literal block quoting.

    Literal:

    {{{
    ||Cell 1||Cell 2||Cell 3||
    ||Cell 4||Cell 5||Cell 6||
    }}}
    
    Display:

    ||Cell 1||Cell 2||Cell 3||
    ||Cell 4||Cell 5||Cell 6||


    Last Line
    """,open_=True, skip=False)


    ts(r"""

    Looks like literals loosing indent ?

    Literal:
    {{{
        Indented text 
    }}}
    
    Display:

        Indented text 

    0123456789

    """,open_=True, skip=True)






