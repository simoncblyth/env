#!/usr/bin/env python
"""
doclite.py
============

Lite weight document model, used for conversion of tracwiki text into RST. 

Refs
----------

* :doc:`/python/python_unicode`


Initial Observations
------------------------

1. seems cannot return unicode from __str__ always returns str (bytes)
2. **overloading __str__ is bad practice when dealing with unicode**


All py2 ascii byte strings get implicitly decoded into unicode 
**assuming that they are ascii** when those str(py2) 
are combined with unicode.



Python2 has __unicode__ for precisely this purpose, return encoded bytes from __str__
----------------------------------------------------------------------------------------

* https://stackoverflow.com/questions/1307014/python-str-versus-unicode


**John Millikin:**

* __str__() is the old method -- it returns bytes
* __unicode__() is the new, preferred method -- it returns characters. 

The names are a bit confusing, but in 2.x we're stuck with them for compatibility reasons. 
Generally, you should put all your string formatting in __unicode__(), and create a stub __str__() method:

::

    def __str__(self):
        return unicode(self).encode('utf-8')

In 3.0, str contains characters, so the same methods are 
named __bytes__() and __str__(). These behave as expected.


Classes
-----------

Lines
    base class list of strings of all the below

Para
    block of text
Literal
    literal block 
Toc
    table of contents
Head
     header text  

Page
    container of content instances of the above

"""

import logging, sys, re, os
log = logging.getLogger(__name__)



U = "".join(map(unichr,range(0xa7,0xff+1)))
#U = u"abc" 
#U = "abc"

assert type(U) is unicode


from env.trac.migration.rsturl import EscapeURL 
from env.trac.migration.inlinetracwiki2rst import InlineTracWiki2RST

EURL = EscapeURL()
INLI = InlineTracWiki2RST()

inline_tracwiki2rst_ = lambda line:INLI(EURL(line)) 
    



class Lines(list):
    def __init__(self, *args, **kwa):
        self.rawlinenos = kwa.pop("rawlinenos", False) 
        list.__init__(self, *args, **kwa)

    def __repr__(self):
        return "<%s %s lines> " % (self.__class__.__name__, len(self))

    def __unicode__(self):
        return "\n".join([repr(self)] + list(self))

    def __str__(self):
        return unicode(self).encode('utf-8')

    def directive(self, name, args, kwa):
        fargs = " ".join(args)
        fqwa = map(lambda _:"   :%s: %s" % (_[0], _[1]), kwa.items() )
        return "\n".join(["",".. %s:: %s" % (name, fargs)] + fqwa + [""] + self.indent(3) + ["",""] )

    def indent(self, n):
        if not self.rawlinenos:
            fmt_ = lambda _:" "*n + _[1]
        else:
            fmt_ = lambda _:"%3d" % (_[0]+1) + " "*n + _[1]
        pass
        return map(fmt_, enumerate(list(self))) 

    def inlined(self):
        return map(inline_tracwiki2rst_, self ) 

    def bullet(self, n):
        return map(lambda _:" "*n + "* " + _, list(self)) 

    def _get_rst(self):
        """placeholder to potentially be overridden"""
        return "\n".join(self)
    rst = property(_get_rst)


class Para(Lines):
    pass 
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
        return "\n".join(["","::", ""] + self.indent(4) + [""] )
    rst = property(_get_rst)


class CodeBlock(Lines):
    def __init__(self, *args, **kwa):
        lang = kwa.pop("lang", "bash")
        linenos = kwa.pop("linenos", False)
        Lines.__init__(self, *args, **kwa)

        log.debug("CodeBlock lang:%s linenos:%s " % (lang, linenos)) 
        self.lang = lang
        self.linenos = linenos

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
        return self.directive("toctree", "", self.kwa )

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

    def _get_rst(self):
        return "\n".join(["","ListTagged(%s):" % self.tags, ""] + self.bullet(0) + [""] )
    rst = property(_get_rst)



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

    def __init__(self, name):
        list.__init__(self)
        self.name = name


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
        return "\n".join(map(lambda _:_.rst, list(self)))
    rst = property(_get_rst)



def dump(obj):
    log.info("repr")
    print repr(obj)
    log.info("str")
    print str(obj)    
    log.info("unicode")
    print unicode(obj)    
    log.info("rst")

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
    log.info("test_para")
    pa = Para( ["red","green","blue", U ] ) 
    dump(pa)
 
def test_lines():
    log.info("test_lines")
    li = Lines( ["red","green","blue", U ] ) 
    dump(li)

def test_Toc():
    toc = Toc(["red","green","blue"], maxdepth=1)
    print toc.rst




if __name__ == '__main__':
    pass
    logging.basicConfig(level=logging.INFO)

    #test_lines() 
    #test_para() 

    test_Toc()


if 0:
    pg = test_page() 
    from env.doc.rstutil import rst2html_open    
    rst = pg.rst
    assert type(rst) is unicode
    rst2html_open(rst, "pg")

    





