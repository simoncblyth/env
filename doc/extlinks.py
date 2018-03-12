#!/usr/bin/env python
"""
::

    ~/e/doc/extlinks.py 

"""
import re, logging, os
log = logging.getLogger(__name__)


class SphinxExtLinks(dict):
    """
    Uses extlink mappings to resolve RST role refs
    into actual urls.
    """
    TRAC_LINK = re.compile("^(?P<typ>\w+)\:(?P<arg>\S*)$")   
    RST_ROLE = re.compile("^\:(?P<typ>\w+)\:\`(?P<arg>\S*)\`$")  
    #DEFAULT_TYP = "wiki" 
    DEFAULT_TYP = "w" 

    LINKREL_TYPS = ["tracdocs", "spx"]


    def __init__(self, *args, **kwa):
        if len(args) == 1 and callable(args[0]):
            func = args[0]
            args = func()
            log.info("calling func %s yields %s extlinks " % (func, len(args)) )
            dict.__init__(self, args, **kwa)
        else:
            dict.__init__(self, *args, **kwa)
        pass

    def resolve(self, txt, docname=None, srcdir=None):
        """
        :param txt: rst interpreted text reference, eg :wiki:`SomePage`
        :param docname: use
        :return url:

        Convert role reference into absolute URL using the Sphinx extlinks mappings. 

        http://localhost/w/sphinxtest/links/sublinks/zh_previous_step.png

        """
        typ, arg_ = self.identify_rst_role(txt)
        if not typ in self:
            return txt 
        pass
        tmpl, pfx = self[typ] 

        if docname is not None and arg_.find("/") == -1:  # looks like a relative link from Sphinx document
            srcname = os.path.basename(srcdir) if srcdir is not None else None
            if srcname == "workflow":  
                srcname = None
            pass
            docdir = os.path.dirname(docname)  # final component of docname is document name not a dir 
            arg = "/".join(filter(None,(srcname, docdir, arg_)))
            log.info("(%s) srcname:%s docname:%s resolved arg   %s -> %s " %  ( typ,srcname, docname,  arg_, arg ))
        else:
            arg = arg_
        pass
        url = tmpl % arg 
        log.info("(%s:%s)[%s] url %s " %  (typ, arg_, [docname,"-"][docname is None], url ))
        return url 

    def __repr__(self):
        return "\n".join(["%10s  :  %100s   : %50s " % ( k, self[k][0], self[k][1] ) for k in self ])

    @classmethod
    def identify_rst_role(cls, txt):
        m = cls.RST_ROLE.match(txt)
        if m is None:
            return None,None
        pass
        typ, arg = m.groups() 
        return typ, arg 


    @classmethod
    def from_traclink(cls, txt):
        m = cls.TRAC_LINK.match(txt)
        if m is None:
            typ, arg = cls.DEFAULT_TYP, txt 
        else: 
            typ, arg = m.groups()
        pass
        xlnk = ":%s:`%s`" % (typ, arg)
        return typ, arg, xlnk

    @classmethod
    def trac2sphinx_link(cls, tlnk, reldoc=None, typ_default="wrel"):
        """
        :param tlnk: trac link of form typ:arg
        :param reldoc:  when defined is typically used to make the link absolute to the trac instance
        :param typ_default:
        """
        if tlnk.find("://") != -1:
            return tlnk   # already absolute  
        pass
        fcolon = tlnk.find(":")
        if fcolon != -1:
            typ = tlnk[:fcolon]
            arg = tlnk[fcolon+1:]
        else:
            typ = typ_default
            arg = tlnk
        pass
        if reldoc is not None:
           arg = "%s/%s" % (reldoc, arg )
        pass

        if typ_default == "bare-relative":
            xlnk = arg
        else:
            xlnk = ":%s:`%s`" % (typ, arg) 
        pass
        return xlnk


    tractable = property(lambda self:"\n".join(map(lambda k:" || %s:something || {{{%s}}} || {{{%s}}} ||  " % (k, self[k][0], self[k][1] ) , self)))

    def __call__(self, a, docname=None, srcdir=None):
        """
        :param a: TracLink (eg wiki:SomePage) or RST role link (eg :wiki:`SomePage`)
        :return url: 
        """
        if a.find("`") != -1:
            url = self.resolve(a, docname=docname, srcdir=srcdir)
            log.debug("(resolve)   %50s -> %s " % ( a, url ))
        else:
            typ, arg, xlnk  = self.from_traclink(a)
            log.debug("(translate) %50s -> %s " % ( a, xlnk ))
            url = self.resolve(xlnk, docname=docname, srcdir=srcdir)
            log.debug("(resolve)   %50s -> %s " % ( xlnk, url ))
        pass
        return url


  


if __name__ == '__main__':

    TRACSRV = "standin.local/tracs/workflow"
    TRACDIR = "/tmp/env/docs/dummy"
    S = "%s"

    SPHINX_EXTLINKS = {
       'tracwiki':('file://%(TRACDIR)s/attachments/wiki/%(S)s' % locals()  , 'tracwiki: '),
       'source':('http://%(TRACSRV)s/browser/%(S)s' % locals() , 'source:'),
       'google':('http://www.google.com/search?q=%s','google:'),
    }
  

    import sys 
    args = sys.argv[1:]
    if len(args) == 0:
        args = [":tracwiki:`FX/fx-oct27-2009.png`",
                ":google:`RST`",
                ":source:`trunk/fullsurvey/survey.py`",
                "google:RST",
                "source:trunk/fullsurvey/survey.py"
                 ]
    pass
    XLNK = SphinxExtLinks(SPHINX_EXTLINKS)
    print "\n".join(map(XLNK,args))


