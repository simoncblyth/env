#!/usr/bin/env python
"""
::

    ~/e/doc/extlinks.py 

"""
import re, logging
log = logging.getLogger(__name__)


class SphinxExtLinks(dict):
    """
    Uses extlink mappings to resolve RST role refs
    into actual urls.
    """
    TRAC_LINK = re.compile("^(?P<typ>\w+)\:(?P<arg>\S+)$")   
    RST_ROLE = re.compile("^\:(?P<typ>\w+)\:\`(?P<arg>\S+)\`$")  


    def __init__(self, *args, **kwa):
        dict.__init__(self, *args, **kwa)

    def resolve(self, txt, docname=None):
        typ, arg = self.identify_rst_role(txt)
        if not typ in self:
            return txt 
        pass
        tmpl, pfx = self[typ] 

        if typ == "tracwiki" and arg.find("/") == -1: 
            if docname is not None:
                arg = "%s/%s" % (docname, arg)
            else:
                log.warning("page relative tracwiki link, but no docname provided")
            pass 
        pass
        url = tmpl % arg 
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
            typ, arg, extlnk = None, None, txt 
        else: 
            typ, arg = m.groups()
            xlnk = ":%s:`%s`" % (typ, arg)
        pass
        return typ, arg, xlnk

    @classmethod
    def trac2sphinx_link(cls, tlnk, typ_default="tracwiki"):
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
        xlnk = ":%s:`%s`" % (typ, arg) 
        return xlnk

    def test(self, args):
        for a in args:
            if a.find("`") != -1:
                url = self.resolve(a)
                print "(resolve)   %50s -> %s " % ( a, url )
            else:
                typ, arg, xlnk  = self.from_traclink(a)
                print "(translate) %50s -> %s " % ( a, xlnk )
                url = self.resolve(xlnk)
                print "(resolve)   %50s -> %s " % ( xlnk, url )
            pass
        pass 




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
    XLNK.test(args)

