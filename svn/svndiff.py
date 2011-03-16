#!/usr/bin/env python
"""
Compare "svn diff" with "svnlook diff" on C2 ... 

  sudo cp -rf  /var/scm/repos/env /tmp/
  sudo chown -R blyth.blyth /tmp/env

  svnlook diff /tmp/env > /tmp/svnlook   
  ( cd ~/env ; svn diff -c $(svnversion))  > /tmp/svndiff
  diff /tmp/svnlook /tmp/svndiff

       * differences in the "chrome" only


  



In [18]: lines[14-1:14+3]
Out[18]: 
['Index: offline/gendbi/tmpl.py',
 '===================================================================',
 '--- offline/gendbi/tmpl.py\t(revision 3009)',
 '+++ offline/gendbi/tmpl.py\t(revision 3010)']

"""

import os, re

hdr = re.compile(r"""
(?P<action>\S*): (?P<path>\S*)
===================================================================
--- (?P<path2>\S*)\S+(?P<adate>.*?)\S+\(rev.* (?P<arev>\d*)\)
\+\+\+ (?P<path3>\S*)\S+(?P<bdate>.*)\S+\(rev.* (?P<brev>\d*)\)
@@ -0,0 +1,94 @@
""")

tst = r"""
Modified: trunk/svn/svnprecommit.bash
===================================================================
--- trunk/svn/svnprecommit.bash 2011-03-16 10:29:06 UTC (rev 3298)
+++ trunk/svn/svnprecommit.bash 2011-03-16 11:53:17 UTC (rev 3299)
@@ -8,7 +8,13 @@
"""


div = "==================================================================="

class SVNDiff(dict):
    _cmd = "svn diff -c %(rev)s"
    cmd = property(lambda self:self._cmd % self)
    def __call__(self, **kwargs):
        self.update(kwargs)
        return os.popen(self.cmd)


if __name__=='__main__':

   #m = hdr.match(tst)

   sd = SVNDiff()
   s = sd(rev=3010).read()
   lines = s.split("\n")
   divs = []
   for i,l in enumerate(lines):
       if l == div:
           divs.append(i)


