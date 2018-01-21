#!/usr/bin/env python
"""
::

   ./test_tracwiki2rst.py 


"""
import logging
log = logging.getLogger(__name__)
from env.trac.migration.resolver import Resolver
from env.doc.rstutil import rst2html_open    
from env.trac.migration.tracwiki2rst import TracWiki2RST

class DummyWikiPage(object):
    metadict = {'tags':"Red Green Blue", 'name':"DummyWikiPage"} 
    name = "MailDebug"
 
class DummyResolver(object):
    def __call__(self, ref, pagename):
        return ref 

class DummyArgs(object):
    origtmpl = None
    origurl = None
    vanilla = True
    rstdir = "/usr/local/workflow/sysadmin/wtracdb/wiki2rst"
    tracdir = "/usr/local/workflow/sysadmin/wtracdb/workflow"


class TestTracWiki2RST(object):
    def __call__(self, txt, open_=False, skip=True):
        if skip:return
        wp = DummyWikiPage()
        ctx = DummyArgs()
        ctx.resolver = Resolver(tracdir=ctx.tracdir, rstdir=ctx.rstdir)

        text = "\n".join(map(lambda _:_[4:], txt.split("\n")[1:-1]))

        pg = TracWiki2RST.page_from_tracwiki(wp, text, ctx)
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


class TestRST(object):
    def __call__(self, txt, open_=False, skip=True, trim=True):
        if skip:return
        if trim:
            text = "\n".join([u""]+map(lambda _:_[4:], txt.split("\n")[1:-1]))
        else:
            text = txt
        pass
        rst = text 
        div = "\n" +  "#" * 100 + "\n"
        print div
        print rst 
        print div
        if open_:
            rst2html_open(rst, "rst")  
        pass



def test_make_index():
    """
    Gives::
 
        Unknown directive type "toctree".

    """
    log.info("test_make_index")

    title = "test_make_index"
    pagenames = "red green blue".split()

    idx = TracWiki2RST.make_index("index", title, pagenames)
    rst = idx.rst

    trst = TestRST()
    trst(rst, open_=True, skip=False, trim=False)
   
   


if __name__ == '__main__':
    #level = 'DEBUG'
    level = 'INFO'
    logging.basicConfig(level=getattr(logging,level))
 
    ts = TestTracWiki2RST()     
    trst = TestRST()     

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

    * note the {{{..}}} RST empty comments placeholders in otherwise empty 1st columns, 
      prior to automating the addition of this tables without had layout messed up 
    
    """,open_=False, skip=True)

    ts(r"""

    Decide that attemping to translate inside literal blocks
    is a can of worms... so simply leave asis, no translation
 
    Literal:

    {{{
    ||Cell 1||Cell 2||Cell 3||
    ||Cell 4||Cell 5||Cell 6||
    }}}
    
    Display:

    ||Cell 1||Cell 2||Cell 3||
    ||Cell 4||Cell 5||Cell 6||


    Last Line
    """,open_=False, skip=True)


    ts(r"""

    Looks like literals loosing indent ?

    Literal:
    {{{
        Indented text 
    }}}
    
    Display:

        Indented text 

    0123456789

    """,open_=False, skip=True)



    ts(r"""


    ||  __DAS Prime__  ||    __DAS Backup__  ||         ||  
    ||  ArchiveA        ||  BackupArchiveA ||  250 ||
    ||  ArchiveB        || BackupArchiveB   ||  250 ||

    Before inlining table cells this gave the below error as Trac underscore 
    misinterpreted as anonymous-hyperlinks. See rst-
    {{{
    System Message: ERROR/3 (/tmp/pg.rst); backlinks: 1, 2
    Anonymous hyperlink mismatch: 2 references but 0 targets. See "backrefs" attribute for IDs.
    }}}

    * http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html#anonymous-hyperlinks

 
    """,open_=False, skip=True)



    ts(r"""

       || '''raw_''' || CF          || '''big dry cabinet''' || '''small dry cabinet''' ||                     ||  
       || 0001:3     ||             ||      blue-12 box      ||                         ||                     ||  
       || 1001:1024  ||             ||      blue-24 box      ||                         ||                     ||  
       || 1025       || CF-00031:33 ||      blue-12 box      ||                         ||                     ||  
       || 1026       || CF-00034:35 ||      blue-12 box      ||                         ||                     ||  
       || 1027       || CF-00036:38 ||      blue-12 box      ||                         ||                     ||    
       || 1028       || CF-00039:40 ||      blue-12 box      ||   purple folder         ||                     ||   
       || 1029       || CF-00041:42 ||  -2-                  ||                         ||                     ||    
       || 1030       || CF-00043:44 ||  -3-                  ||                         ||                     ||    
       || 1031       || CF-00045:47 ||  -4-                  ||                         ||                         ||  
       || 1032       || CF-00048:49 ||  -5-                  ||                         ||  End 2009/2010 jan      ||  
       || 1033       || CF-00050:51 ||  -6-                  ||                         ||                         ||  


    """,open_=False, skip=True)


    ts(r"""


    [[Image(EnterPassword.png)]] 


    """, open_=False, skip=True)


    trst(r"""

    * http://docutils.sourceforge.net/docs/ref/rst/roles.html#title-reference

    * :strong:`strong`
    * :emphasis:`emphasis`
    * :subscript:`subscript`
    * :superscript:`superscript`


    ref gives **Unknown interpreted text role "ref"** as that is a Sphinx addition.

    ::

        * :ref:`genindex`

    :tail: tail
    :tail2: tail2

    """, open_=False, skip=True)


    ts(r"""

    === 1928 route map === 

     * http://www.airbus.com.tw/images/roadmap/1968_1060119.jpg

    [[Image(http://www.airbus.com.tw/images/roadmap/1968_1060119.jpg)]]

     Any indent getting thru to RST on the bullet line following an image gives
    {{{
    Error in "image" directive: no content permitted.
    }}}

    """, open_=False, skip=True)



    ts(r"""
    = Indented bullet list after literal block = 

    Anything indented following the literal block gets 
    incorporated into the literal block. Tried to add an
    RST comment, this then causes the indented block to get commented.

    Hmm seems unavoidable, I have to take control of the indent, and 
    offset it to zero 

    {{{
    #!py
           def _DayaBay__HepMCEvent(self):
                 return "<%s  generatorName:%s >" % ( self.__class__.__name__ , self.generatorName() ) 

           kls=PyCintex.makeClass('DayaBay::HepMCEvent')
           kls.__repr__ = _DayaBay__HepMCEvent
    }}}
       
    * source:/trunk/dybpy/dybpy/
    * source:/trunk/dybpy/dybpy/genrepr.py
    * source:/trunk/dybpy/dybpy/geneventlook.py     

    """, open_=True, skip=False)


    #test_make_index()


