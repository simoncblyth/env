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



class Lines(list):
    def __init__(self, *args, **kwa):
        list.__init__(self, *args, **kwa)

    def __repr__(self):
        return "<%s %s lines> " % (self.__class__.__name__, len(self))

    def __unicode__(self):
        return "\n".join([repr(self)] + list(self))

    def __str__(self):
        return unicode(self).encode('utf-8')

    def indent(self, n):
        return map(lambda _:" "*n + _, list(self)) 

    def bullet(self, n):
        return map(lambda _:" "*n + "* " + _, list(self)) 

    def _get_rst(self):
        """placeholder to potentially be overridden"""
        return "\n".join(self)
    rst = property(_get_rst)


class Para(Lines):
    pass 

class Literal(Lines):
    start = "{{{"
    end = "}}}"

    @classmethod 
    def is_start(cls, line):
        return line.startswith(cls.start)
    @classmethod 
    def is_end(cls, line):
        return line.startswith(cls.end)

    def _get_rst(self):
        return "\n".join(["","::", ""] + self.indent(4) + [""] )
    rst = property(_get_rst)


class Toc(Lines):     
    def _get_rst(self):
        return "\n".join(["",".. toctree::", ""] + self.indent(3) + ["",""] )
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

  

class Head(Lines):
    """
    http://docutils.sourceforge.net/docs/user/rst/quickref.html#section-structure
    """
    ptn = re.compile("^(=+) ([^=]*) (=+)$")
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
        return "<Head %s %s %s lines> " % (self.title, self.level, len(self))

    @classmethod 
    def is_match(cls, line):
        m = cls.ptn.match(line)
        return m is not None

    @classmethod 
    def match(cls, line):
        m = cls.ptn.match(line)
        assert m , "match failed for line [%s] " % line
        a, title, b = m.groups()

        if len(a) != len(b):
           log.warning("got unequal a, b  %s %s for header: %s " % (a,b, line))
        pass
        level = max(len(a), len(b))

        if int(level)-1 > len(cls.mkr):
            log.fatal("Head level %d too large for line [%s]  " % (level, line ))
            assert 0   
        pass
        return title, level

    @classmethod
    def from_line(cls, line):
        assert cls.is_match(line)
        title, level = cls.match(line) 
        head = cls(title, level, line=line)
        head.append(line)
        return head

    def __init__(self, title, level, line=None):
        self.title = title
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
    pg.append(Head("Demo SubTitle Second", 2))
    pg.append(Para( ["red2","green2","blue2"]  ))
    pg.append(Literal( ["red2","green2","blue2"]  ))

    assert pg.title == "Demo Title" 


    dump(pg)

def test_para():
    log.info("test_para")
    pa = Para( ["red","green","blue", U ] ) 
    dump(pa)
 
def test_lines():
    log.info("test_lines")
    li = Lines( ["red","green","blue", U ] ) 
    dump(li)

if __name__ == '__main__':
    pass
    logging.basicConfig(level=logging.INFO)

    #test_lines() 
    #test_para() 
    test_page() 
    





