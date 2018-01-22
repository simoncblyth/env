#!/usr/bin/env python
"""
doclite.py
============

Lite weight document model, used for conversion of tracwiki text into RST. 

Refs
----------

* :doc:`/python/python_unicode`

Classes
-----------

Lines
    base class list of strings of all the below

Para
    block of text
Literal
    literal block 
SimpleTable
    produces non-grid RST table from simple Trac one
CodeBlock
    literal code with lang syntax coloring 
Toc
    table of contents
ListTagged
    an incomplete element of the page, filled in later up where there is DB access
Meta
    field list from a dict  
Sidebar
    some metadata off to side
Contents
    in body contents, showing list of section titles
Anchor
    Sphinx index anchor using tags and the page name
HorizontalRule
    transition line 
Head
     header text  

Page
    container of content instances of the above

"""

import logging, sys, re, os
log = logging.getLogger(__name__)

U = "".join(map(unichr,range(0xa7,0xff+1)))
assert type(U) is unicode

import copy 
from env.trac.migration.inlinetracwiki2rst import inline_tracwiki2rst_, TableTracWiki2RST



class Lines(list):
    def __init__(self, *args, **kwa):
        fmt = kwa.pop('fmt', "tracwiki")
       
        list.__init__(self, *args, **kwa)
        #log.info("Lines %s fmt:%s " % (self.__class__.__name__, fmt) )
        self.fmt = fmt
     

    def __repr__(self):
        return "<%s %s lines> " % (self.__class__.__name__, len(self))

    def __unicode__(self):
        return "\n".join([repr(self)] + list(self))

    def __str__(self):
        return unicode(self).encode('utf-8')

    def directive(self, name, args, **kwa):
        tail = kwa.pop("tail", [])
        fargs = " ".join(args)
        fqwa = map(lambda _:"   :%s: %s" % (_[0], _[1]), kwa.items() )
        return "\n".join(["",".. %s:: %s" % (name, fargs)] + fqwa + [""] + self.indent(3) + [""] + tail )

    def indent(self, n):
        fmt_ = lambda _:" "*n + _
        return map(fmt_, self) 

    def inlined(self):
        inline_ = inline_tracwiki2rst_ if self.fmt == "tracwiki" else lambda _:_
        return map(inline_, self ) 

    def bullet(self, n):
        return map(lambda _:" "*n + "* " + _, list(self)) 

    def _get_rst(self):
        """placeholder to potentially be overridden"""
        return "\n".join(self)
    rst = property(_get_rst)


class Para(Lines):
    def __init__(self, *args, **kwa):
        Lines.__init__(self, *args, **kwa)

    def _get_rst(self):
        return "\n".join(self.inlined())
    rst = property(_get_rst)


class Literal(Lines):
    """
    * permit indented literal, eg as used on WikiRestructuredText
      this page also has nested literal blocks : which are not worthy of supporting

    """
    start = "{{{"
    end = "}}}"

    def __init__(self, *args, **kwa):
        Lines.__init__(self, *args, **kwa)

    @classmethod 
    def is_start(cls, line):
        return line.lstrip().startswith(cls.start)
    @classmethod 
    def is_end(cls, line):
        return line.lstrip().startswith(cls.end)

    def _get_rst(self):
        comment = ["..", "   end-literal ", "" ]
        if len(self) == 0:
            return None
        else:
            return "\n".join(["","::", ""] + self.indent(4) + [""] + comment )
        pass
    rst = property(_get_rst)


class SimpleTable(Lines):
    TABLE_ROW_TOKEN = "||"

    def __init__(self, *args, **kwa):
        self.pagename = kwa.pop('pagename', None)
        self.inline_ = inline_tracwiki2rst_ if kwa.pop('inline', False) else None
        self.conv = TableTracWiki2RST() 
        Lines.__init__(self, *args, **kwa)

    def _get_rst(self):
        map(self.conv, self)  ## collects possibly inline converted tracwiki text cells into list of lists 

        tab = self.conv._table
        tab.apply_func(self.inline_)  ## inline replacements 
 
        topleftcell = tab[0][0].lstrip()

        if len(topleftcell) > 2 and topleftcell[0:2] == "**":  # heuristic 
            tab.hdr = True
        pass 

        try:
            urst = unicode(tab)
        except AssertionError as err:
            log.fatal("SimpleTable caught assert for page %s " % self.pagename)
            log.fatal(" err : %s " % err )
            sys.exit(1)
        pass
        self[:] = urst.split("\n")
        return "\n".join(self)

    rst = property(_get_rst)

    @classmethod 
    def is_simpletable(cls, line):
        return line.lstrip().startswith(cls.TABLE_ROW_TOKEN)
     

class CodeBlock(Lines):
    def __init__(self, *args, **kwa):
        lang = kwa.pop("lang", "bash")
        vanilla = kwa.pop("vanilla", False)
        linenos = kwa.pop("linenos", False)
        rawlinenos = kwa.pop("rawlinenos", False) 

        if vanilla and linenos:
            log.warning("CodeBlock linenos is a Sphinx extension, switching off as vanilla RST is selected ")
            linenos = False
        pass

        Lines.__init__(self, *args, **kwa)

        log.debug("CodeBlock lang:%s linenos:%s vanilla:%s " % (lang, linenos, vanilla)) 
        self.lang = lang
        self.linenos = linenos
        self.rawlinenos = rawlinenos
        self.vanilla = vanilla

    def indent(self, n):
        if not self.rawlinenos:
            fmt_ = lambda _:" "*n + _[1]
        else:
            fmt_ = lambda _:"%3d" % (_[0]+1) + " "*n + _[1]
        pass
        return map(fmt_, enumerate(list(self))) 

    def _get_rst(self):
        pst = ["   :linenos:",""] if self.linenos else ["" ]
        return "\n".join(["",".. code-block:: %s" % self.lang] + pst + self.indent(4) + [""] )
    rst = property(_get_rst)

    def __repr__(self):
         return "<%s lang::%s linenos:%s lines:%s>" % (self.__class__.__name__, self.lang, self.linenos, len(self)) 
        

class Toc(Lines):     
    def __init__(self, *args, **kwa):
        self.kwa = kwa
        Lines.__init__(self, *args)

    def _get_rst(self):
        #return "\n".join(["",".. toctree::", ""] + self.indent(3) + ["",""] )
        return self.directive("toctree", [], **self.kwa )

    rst = property(_get_rst)


class ListTagged(Lines):

    ptn = re.compile("\[\[ListTagged\(([^\)]*)\)\]\]") 

    @classmethod 
    def is_match(cls, line):
        m = cls.ptn.match(line)
        return m is not None

    @classmethod 
    def match(cls, line):
        m = cls.ptn.match(line)
        assert m , "match failed for line [%s] " % line
        tags, = m.groups()
        return tags

    @classmethod
    def from_line(cls, line):
        assert cls.is_match(line)
        tags = cls.match(line) 
        tgls = cls(tags )
        return tgls

    def __init__(self, tags):
        self.tags = tags
        Lines.__init__(self)

    def _get_rst(self):
        return "\n".join(["","ListTagged(%s):" % self.tags, ""] + self.bullet(0) + [""] )
    rst = property(_get_rst)


class Image(Lines):
    ptn = re.compile("\[\[Image\(([^\)]*)\)\]\]") 

    @classmethod 
    def is_match(cls, line):
        """
        Uses kludgy way to avoid matching literal-ized Image macro
        """
        m = cls.ptn.search(line)
        if m is None:
            return False 
        pass
        lhs = line[:m.start()]
        if lhs.find("`") > -1 or lhs.find("{{{") > -1:
            return False 
        pass 
        return True

    @classmethod 
    def match(cls, line):
        m = cls.ptn.search(line)
        assert m is not None
        refs = m.groups()
        assert len(refs) == 1
        return refs[0]

    @classmethod
    def from_line(cls, line, ctx, docname=None):
        """
        Note that resolving of links is left to the Sphinx/sphinxext machinery,
        this just re-formats the link layout to sphinx form. 
        """
        assert cls.is_match(line)
        tlnk = cls.match(line) 

        xlnk = ctx.extlinks.trac2sphinx_link(tlnk, typ_default="wikidocs")

        log.info("Image.from_line  translating tlnk to xlnk %s -> %s  (%s) " % (tlnk, xlnk, docname))

        img = cls(xlnk, docname)
        return img

    def __init__(self, url, docname):
        self.url = url
        self.docname = docname
        Lines.__init__(self)

    def _get_rst(self):
        """
        The comment after the image avoids error from following indented 
        content causing a "no content permitted" error.
        """
        comment = ["..", "   image url:%s docname:%s  " % (self.url, self.docname), "" ]
        #dr = "image"
        dr = "wimg"
        return self.directive(dr, [self.url], tail=comment) 

    rst = property(_get_rst)

    def __repr__(self):
        return "<Image %s> " % (self.url)



class Meta(Lines):
    def __init__(self, md):
        Lines.__init__(self)
        self.md = md

    def _get_rst(self):
        return "\n".join([""]+[":%s: %s" % ( k, v ) for k,v in self.md.items()]+self.indent(0)+[""])
    rst = property(_get_rst)

    def __repr__(self):
        return "<Meta %s> " % (self.md)


class Sidebar(Lines):

    M = {"ftime":"Date", "author":"Authors" } 
    def __init__(self, md):
        Lines.__init__(self)
        self.md = md
    def _get_rst(self):
        return "\n".join([""]+[".. sidebar:: %s" % self.md["name"], ""] + [ "   :%s: %s" % (self.M[k],v) for k,v in self.md.items() if k in self.M]+[""])
    rst = property(_get_rst)

    def __repr__(self):
        return "<Sidebar %s> " % (self.md)

class Contents(Lines):
    def __init__(self, depth):
        Lines.__init__(self)
        self.depth = depth
    def _get_rst(self):
        return "\n".join([""]+[".. contents::"] + [ "   :depth: %s" % self.depth ]+[""])
    rst = property(_get_rst)

    def __repr__(self):
        return "<Contents %s> " % (self.depth)

      
class Anchor(Lines):
    """
    See w-:SOP/sphinxuse.rst
    """
    def __init__(self, name, tags):
        Lines.__init__(self)
        self.tags = list(set(tags.split() + [name])) 

    def _get_rst(self):
        fidx = ", ".join(self.tags)
        return "\n".join([""]+[".. index:: %s" % fidx ]+[""])
    rst = property(_get_rst)

    def __repr__(self):
        return "<Anchor %s> " % (self.tags)



class HorizontalRule(Lines):
    ptn = re.compile("^-{4,100}\s*$")
    @classmethod 
    def is_match(cls, line):
        m = cls.ptn.match(line)
        return m is not None

    def __init__(self):
        Lines.__init__(self, ["","----", ""] )



class Head(Lines):
    """
    http://docutils.sourceforge.net/docs/user/rst/quickref.html#section-structure

    g4pb:/usr/local/env/trac/package/tractrac/trac-0.11/trac/wiki/parser.py 

    ""

        In [16]: re.compile("(?P<heading>^\s*(?P<hdepth>=+)\s*(?P<title>.*?)\s*(?P=hdepth)\s*)").match(line).groupdict()
        Out[16]: 
        {'hdepth': '===',
         'heading': '=== --log=error : smartctl -d ata -l error /dev/hda === ',
         'title': '--log=error : smartctl -d ata -l error /dev/hda'}


    """
    #ptn = re.compile("^(=+)\s*([^=]*)\s*(=+)\s*$")  ## NOPE some titles contain "="
    ptn = re.compile("(?P<heading>^\s*(?P<hdepth>=+)\s*(?P<title>.*?)\s*(?P=hdepth)\s*)")
 
    mkr = list("=-~+#<>")   

    def _get_rst(self):
        level = int(self.level)-1
        if level >= len(self.mkr):
            log.warning("Head level %d too large for %r  " % (level, self) ) 
            level = len(self.mkr)-1
        pass
        return "\n".join(["", self.title, self.mkr[level] * len(self.title), "" ])
    rst = property(_get_rst)

    def __repr__(self):
        """ 
        including title in repr is problematic, as might contain non-ascii 
        """
        title = unicode(self.rawtitle)
        btitle = title.encode("ascii", "replace")
        return "<Head %s %s %s lines> " % (btitle, self.level, len(self))

    @classmethod 
    def is_match(cls, line):
        m = cls.ptn.match(line)
        return m is not None

    @classmethod 
    def match(cls, line, name=None):
        m = cls.ptn.match(line)
        assert m , "match failed for line [%s] " % line
        d = m.groupdict()
        title = d["title"]
        level = len(d["hdepth"])

        if int(level)-1 > len(cls.mkr):
            log.fatal("Head level %d too large for line [%s]  " % (level, line ))
            assert 0   
        pass
        return title.strip(), level

    @classmethod
    def from_line(cls, line, name=None):
        assert cls.is_match(line)
        title, level = cls.match(line, name=name) 
        head = cls(title, level, line=line)
        head.append(line)
        return head

    def __init__(self, title, level, line=None):
        Lines.__init__(self)
        self.rawtitle = title
        self.title = inline_tracwiki2rst_(title)
        self.level = level
        if line is None:
            mk = "f" * level 
            fab = "%s %s %s" % (mk,title,mk)
            line = fab 
        pass 
        self.append(line)



class Page(list):
    """
    Page
    =====

    A container of content in the form of a list of instances of:

    * Para
    * Head
    * Literal 

    """
    INCOMPLETE = [ListTagged]

    def __init__(self, *args, **kwa):
        name = kwa.pop('name',None)
        assert name is not None
        list.__init__(self, *args, **kwa)
        self.name = name

    def findall(self, cls):
        if type(cls) is str:  
            return filter(lambda _:type(_).__name__ is cls, self)
        else:
            return filter(lambda _:type(_) is cls, self)
        pass

    def count(self, cls):
        return len(self.findall(cls))

    def incomplete_instances(self):
        return filter(lambda _:type(_) in self.INCOMPLETE, self)

    def _get_title(self):
        """first Head title, or name if no Head"""
        for _ in self:
            if type(_) is Head:
                return _.title
            pass
        return self.name
    title = property(_get_title)

    def __repr__(self):
        return "\n".join(map(repr, list(self)))

    def __str__(self):
        return "\n".join(map(str, list(self)))

    def __unicode__(self):
        return "\n".join(map(unicode, list(self)))
       
    def _get_rst(self):
        return "\n".join(filter(lambda _:_ is not None, map(lambda _:_.rst, list(self))))
    rst = property(_get_rst)




def banner(msg):
    print "#" * 50 + " %-30s " % msg + "#" * 50 

def dump(obj):
    banner("repr(obj)")
    print repr(obj)
    banner("str(obj)")
    print str(obj)    
    banner("unicode(obj)")
    print unicode(obj)    
    banner("obj.rst")
    rst = obj.rst
    assert type(rst) is unicode
    print rst    
  

def test_page():
    log.info("test_page")
    pg = Page("demo")
    pg.append(Head("Demo Title", 1))
    pg.append(Head("Demo SubTitle First", 2))
    pg.append(Para( ["red","green","blue", U ]  ))
    pg.append(Literal( ["red","green","blue"]  ))
    pg.append(HorizontalRule())
    pg.append(Head("Demo SubTitle Second", 2))
    pg.append(Para( ["red2","green2","blue2"]  ))
    pg.append(Literal( ["red2","green2","blue2"]  ))

    pg.append(Para( [""," * http://www.google.com/some/funny_path_", " * oyster "]  ))

    assert pg.title == "Demo Title" 
    dump(pg)
    return pg  


def test_para():
    """
    Using fmt=rst prevents the tracwiki2rst inlining 
    """
    log.info("test_para")
    pa = Para( ["red","green","blue", U, ":strong:`strong`" ], fmt="rst") 
    dump(pa)
  
def test_lines():
    log.info("test_lines")
    li = Lines( ["red","green","blue", U, ":strong:`strong`" ], fmt="rst" ) 
    dump(li)

def test_Toc():
    toc = Toc(["red","green","blue"], maxdepth=1)
    print toc.rst




if __name__ == '__main__':
    pass
    logging.basicConfig(level=logging.INFO)

    #test_lines() 
    test_para() 

    #test_Toc()


if 0:
    pg = test_page() 
    from env.doc.rstutil import rst2html_open    
    rst = pg.rst
    assert type(rst) is unicode
    rst2html_open(rst, "pg")

    





