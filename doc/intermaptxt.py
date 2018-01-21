#!/usr/bin/env python
"""
intermaptxt.py
===============

Parses the Trac InterMapTxt.txt file, allowing the 
Sphinx extlink equivalent to be obtained.

InterMapTxt.txt::

    db  https://wiki.bnl.gov/dayabay/index.php?title=$1                 # BNL dayabay public wiki page $1   formerly mwpub
    zh  http://translate.google.ca/translate?hl=en&sl=zh&u=$1    # google translate from Chinese to English


Note that InterMapTxt supports multiple substitutions whereas Sphinx extlink
supports only one, so such links are broken.
"""
import os, re, logging
log = logging.getLogger(__name__)
from collections import OrderedDict


class InterMapTxt(OrderedDict):
     PTN = re.compile("^(?P<label>[a-zA-Z0-9_-]+)\s*(?P<tmpl>\S*://\S*)\s*(?P<tail>.*)$")
     FMT = "%(idx)2d : %(label)20s : %(tmpl)100s : %(tail).50s " 
     QDC = ["%3A%3Fq%3D", ":?q="]   # urlencoded/decoded  
  

     @classmethod
     def from_parse(cls, path=None):
         if path is None:
             path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "InterMapTxt.txt")
         pass
         imt = cls()
         if not os.path.exists(path):
             log.warning("InterMapTxt non-existing path %s " % path) 
             return imt 
         pass
         for line in open(path,"r").readlines():
             match = cls.PTN.match(line)
             if not match:continue
             d = match.groupdict()
             d['idx'] = len(imt)
             imt[d["label"]] = d 
         pass
         imt.fixup()
         log.info("parsing %s found %s mappings " % (path, len(imt)))
         return imt

     def __init__(self, *args, **kwa):
         OrderedDict.__init__(self, *args, **kwa)

     def fixup(self):
         for k in self:
             tmpl = self[k]["tmpl"]
             if tmpl.find("$1") == -1:
                 tmpl = tmpl + "%s"
             else:
                 tmpl = tmpl.replace("$1", "%s")
             pass
             if tmpl.find("$2") != -1:  ## link breaking kludge
                 tmpl = tmpl.replace("$2", "")
             pass

             if tmpl.find(self.QDC[0]) != -1:
                 #tmpl = tmpl.replace(self.QDC[0], self.QDC[1])
                 tmpl = tmpl.replace(self.QDC[0], "")  # dont care about broken links, but do need to satify Sphinx
             pass
             self[k]["tmpl"] = tmpl
         pass

     def _get_extlinks(self):
         return OrderedDict(zip(self.keys(),[(self[k]["tmpl"],"%s:"%k) for k in self.keys()]))
     extlinks = property(_get_extlinks)

     def __str__(self):
         return "\n".join(map(lambda k:self.FMT % self[k], self))



if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO)
    path = sys.argv[1] if len(sys.argv)>1 else None
    imt = InterMapTxt.from_parse(path)
    print imt 
    print imt.extlinks
     




