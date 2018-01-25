#!/usr/bin/env python
"""

"""
import logging, sys, re, os, collections, datetime, codecs, copy
log = logging.getLogger(__name__)

from env.trac.migration.doclite import Para, Head, HorizontalRule, ListTagged, Toc, Literal, CodeBlock, Meta
from env.trac.migration.doclite import Anchor, Contents, Sidebar, Page, SimpleTable, Image
from env.trac.migration.ls import LS
from env.trac.migration.bulletspacer import BulletSpacer

       
class TracWiki2RST(object):
    """
    """ 

    skips = r"""
    [[TracNav
    [[PageOutline
    [[TOC
    """.lstrip().rstrip().split()


    @classmethod
    def meta_top(cls, wp, args):

        out = []
        md = wp.metadict

        if not args.vanilla:
            anchor = Anchor(name=wp.name, tags=md["tags"])
            out.append(anchor)
            sidebar = Sidebar(md=md)
            out.append(sidebar) 
        pass

        contents = Contents(depth=2)
        out.append(contents)

        meta = Meta(md=md)
        meta.append(":orphan:")

        if args.origtmpl is not None:
            origurl = args.origtmpl % wp.name 
            editurl = "%s?action=edit" % origurl 
            meta.append(":origurl: %s" % origurl)
            meta.append(":editurl: %s" % editurl)
        pass
        meta.append(u"")  # ensure some unicode, to kick in coercion
        out.append(meta)

        return out 

     
    @classmethod
    def dbg_tail(cls, wp, text, page):
        """
        This would be wrong (fails with non-ascii) as it mixes byte strings and unicode::

           pg.append(Literal(str(pg).split("\n"))) 

        """
        page0 = copy.deepcopy(page) # avoid including these debug additions in the dump 

        page.add(Head("%s dbg_tail" % wp.name,level=1))

        page.add(Head("Literal converted rst",level=2))
        page.add(CodeBlock(pg0.rst.split("\n"), lang="rst", linenos=True))

        page.add(Head("Literal tracwiki text",level=2))
        page.add(CodeBlock(text.split("\n"),lang="bash", linenos=True)) 
        # NB not wp.text as need to obey .txt file overrides of DB content, see wtracdb-edtest

        page.add(Head("Literal repr(pg)",level=2))
        page.add(CodeBlock(repr(page0).split("\n"), lang="pycon", linenos=True))
        ## repr always fits in ascii ?

        page.add(Head("Literal unicode(pg)",level=2))
        page.add(CodeBlock(unicode(page0).split("\n"), lang="pycon", linenos=True))

    @classmethod
    def page_from_tracwiki(cls, wp, text, ctx, dbg=False):
        name = wp.name
        log.debug("page_from_tracwiki %s " % name)

        mtop = cls.meta_top(wp, ctx) 

        try:
            s_text = BulletSpacer.spaced_out(text)
        except AssertionError:
            log.fatal("caught assertion in BulletSpacer for page %s " % name)
            sys.exit(1)
        pass

        ls = LS(s_text, skips=cls.skips)

        pg = Page(name=name, ctx=ctx, ls=ls)

        for _ in mtop:
            pg.add(_)
        pass

        conv = cls(pg)
        #print ls 

        ## if page lacks a Header, insert one after metadata
        if pg.count(Head) == 0:   
            head =  Head(name, level=1, ctx=ctx)
            head.page = pg  # normally done by add
            pg.insert( len(mtop), head)
        pass

        ## ListTagged requires db access so done here
        for lti in pg.incomplete_instances():
            wp.complete_ListTagged(lti, skipdoc=ctx.skipdoc)
        pass

        if dbg and not ctx.vanilla:
            cls.dbg_tail(wp, text, pg) 
        pass
        return pg 

    @classmethod
    def make_index(cls, name, title, pagenames, ctx):
        idx = Page(name=name)        
        hdr = Head(title, level=1, ctx=ctx)
        toc = Toc( pagenames, maxdepth=1 )

        foot = Head(u"indices and tables", level=1, ctx=ctx)

        para = Para(fmt="rst", ctx=ctx)   # specifying rst prevents tracwiki2rst inlining/escaping etc..
        para.append(u"")         # some unicode is needed, to get the py2 coercion to kick in 
        para.append("* :ref:`genindex`")
        para.append("* :ref:`modindex`")
        para.append("* :strong:`strong`")
        para.append("")

        idx.append(hdr) 
        idx.append(toc) 
        idx.append(foot) 
        idx.append(para) 

        return idx

    def end_para(self):
        if self.cur_para is None:return
        self.page.add(self.cur_para)
        self.cur_para = None
    def end_literal(self, l):
        if self.cur_literal is None:return

        self.page.add(self.cur_literal)
        self.last_literal = self.cur_literal

        if self.cur_literal.in_ != l['indent']:
            print self.page.ls
            log.fatal("%s:%s literal with inconsistent indents %s %s line %s " % (self.name, l['idx'], self.cur_literal.in_, l['indent'], l['line'] )) 
            assert self.cur_literal.in_ == l['indent'], (self.cur_literal.in_, l['indent'], l['line'], l['idx'] ) 
        pass
        self.cur_literal = None


    def add_line(self, l):
        if self.cur_para is None:
             self.cur_para = Para(ctx=self.ctx)
        pass
        self.cur_para.append(l) 

    def __init__(self, page ):
        self.page = page
        self.ctx = page.ctx
        self.name = page.name

        self.cur_para = None
        self.cur_literal = None
        self.last_literal = None
        self.cur_table = None
        self.postliteral = False

        ls = self.page.ls

        for l in ls:
            self.ctx.l = l 
            if l['skip']: 
                l['kls'] = 'Skip' 
                continue

            line = l['line']

            if self.cur_literal is None and Head.is_match(l):  
                self.end_para()
                head = Head.from_line(l, name=self.name, ctx=self.ctx)
                self.page.add(head) 
                l['kls'] = 'Head' 
                self.last_literal = None   # sectioning from Head avoids problem of postliteral indent greediness
            elif self.cur_literal is None and Image.is_match(l):  
                self.end_para()
                img = Image.from_line(l, docname=self.name, ctx=self.ctx)
                self.page.add(img) 
                l['kls'] = 'Image' 
            elif self.cur_literal is None and HorizontalRule.is_match(l):
                self.end_para()
                hr = HorizontalRule(ctx=self.ctx)
                self.page.add(hr) 
                l['kls'] = 'HorizontalRule' 
            elif self.cur_literal is None and ListTagged.is_match(l):
                self.end_para()
                tgls = ListTagged.from_line(l, ctx=self.ctx)
                self.page.add(tgls) 
                l['kls'] = 'ListTagged' 
            elif self.cur_literal is None and SimpleTable.is_simpletable(l):
                self.end_para()
                if self.cur_table is None:
                    l['kls'] = 'SimpleTable.Start' 
                    self.cur_table = SimpleTable(pagename=self.name,inline=True, ctx=self.ctx) 
                else:
                    l['kls'] = 'SimpleTable.Ctd' 
                pass
                self.cur_table.append(line)
            elif Literal.is_start(l):
                self.end_para()
                self.cur_literal = Literal(ctx=self.ctx, in_=l['indent']) 
                l['kls'] = 'Literal.Start' 
            elif Literal.is_end(l):
                self.end_literal(l)
                l['kls'] = 'Literal.End' 
            elif self.cur_table is not None:  # have just left the table 
                self.page.add(self.cur_table)
                self.cur_table = None 
                self.add_line(l)   # avoid chomping the blank line following tables
                l['kls'] = 'LineAfterTable' 
            else:
                if self.cur_literal is not None:
                    l['kls'] = 'Literal.Ctd' 
                    self.cur_literal.append(line)
                else:
                    self.add_line(l)
                    l['kls'] = 'Para' 
                    if self.last_literal is not None and not len(line.strip()) == 0:  # first non-blank line after last literal
                        l['kls'] = "Para.postliteral" 
                        if l['indent'] > self.last_literal.in_: 
                            self.ctx.stats["fix_postliteral_indent"] += 1 
                            msg = "%s.txt:%d postliteral indent %s > %s line [%s] " 
                            log.debug(msg % ( self.name, l['idx'], l['indent'], self.last_literal.in_, l['line'] ))
                            l['offset'] = self.last_literal.in_ - l['indent']
                            ls._sli = slice( max(0, l['idx']-10), min( l['idx']+10, len(ls) ), 1 )
                            #print ls
                        pass
                        self.last_literal = None
                    pass      
                pass 
            pass
        pass
        self.end_para()


       


if __name__ == '__main__':
    pass
    print "see test_tracwiki2rst.py"

