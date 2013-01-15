#!/usr/bin/env python
"""
Mechanized Shiftcheck
=======================

::

   ./shiftcheck.py -n 6          # for the hourly check       
   ./shiftcheck.py -n 1000       # for the 4-hour check

   ./shiftcheck.py -n 6 -o /tmp/1hr    # convenient to write 1hr PNGs in separate folder

   open /tmp/1hr          # coverflow in Finder is sufficient for 6 PNGs

   open file:///tmp/env/web/dayawane.ihep.ac.cn/twiki/bin/view/Internal/ShiftCheck/annotated.html    

        # make sure to refer to the annotated page corresponding to the shiftcheck run to avoid confusion
        # arising from twiki updates 

1hr Check
----------

::

    simon:web blyth$ ./shiftcheck.py -n 6 -o /tmp/1hr
    INFO:__main__:retreiving into pre-existing dir /tmp/1hr 
    INFO:shiftcheck:. 001_EH1__Temperature.png                                      http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=DBNS_PTH_T1&ParaNames%5B%5D=DBNS_PTH_T2 ...
    INFO:shiftcheck:. 002_EH1__Humidity.png                                         http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=DBNS_PTH_H1&ParaNames%5B%5D=DBNS_PTH_H2 ...
    INFO:shiftcheck:. 003_EH2__Temperature.png                                      http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=LANS_PTH_T1&ParaNames%5B%5D=LANS_PTH_T2 ...
    INFO:shiftcheck:. 004_EH2__Humidity.png                                         http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=LANS_PTH_H1&ParaNames%5B%5D=LANS_PTH_H2 ...
    INFO:shiftcheck:. 005_EH3__Temperature.png                                      http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=FARS_PTH_T1&ParaNames%5B%5D=FARS_PTH_T2 ...
    INFO:shiftcheck:. 006_EH3__Humidity.png                                         http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=FARS_PTH_H1&ParaNames%5B%5D=FARS_PTH_H2 ...
    INFO:__main__:STAT
    {1: '.', 2: '.', 3: '.', 4: '.', 5: '.', 6: '.'}

    INFO:__main__:wrote annotated target page to /private/tmp/1hr/annotated.html 
    simon:web blyth$ open /private/tmp/1hr/annotated.html                               # ShiftCheck with links indexed corresponding to above PNG name


4hr Check
-----------

::

    simon:web blyth$ time ./shiftcheck.py -n 1000 -o /tmp/4hr
    INFO:__main__:creating output dir /tmp/4hr 
    INFO:shiftcheck:. 001_EH1__Temperature.png                                      http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=DBNS_PTH_T1&ParaNames%5B%5D=DBNS_PTH_T2 ...
    INFO:shiftcheck:. 002_EH1__Humidity.png                                         http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=DBNS_PTH_H1&ParaNames%5B%5D=DBNS_PTH_H2 ...
    INFO:shiftcheck:. 003_EH2__Temperature.png                                      http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=LANS_PTH_T1&ParaNames%5B%5D=LANS_PTH_T2 ...
    INFO:shiftcheck:. 004_EH2__Humidity.png                                         http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=LANS_PTH_H1&ParaNames%5B%5D=LANS_PTH_H2 ...
    ...
    INFO:shiftcheck:. 238_EH3__Weights.png                                          http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=weight_isobutane&ParaNames%5B%5D=weight ...
    INFO:shiftcheck:. 239_EH3__Pressures.png                                        http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=pressure_argon&ParaNames%5B%5D=pressure ...
    INFO:shiftcheck:. 240_EH1_EH1_RPC_VME_Temperature.png                           http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=FanTemperature&ParaNames%5B%5D=Temperat ...
    INFO:shiftcheck:. 241_EH2_EH2_RPC_VME_Temperature.png                           http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=FanTemperature&ParaNames%5B%5D=Temperat ...
    INFO:shiftcheck:. 242_EH3_EH3_RPC_VME_Temperature.png                           http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=FanTemperature&TimeSpan=1440&Interval=1 ...
    INFO:shiftcheck:. 243_SAB_Temperature___Last_30_Minutes.png                     http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=DBNS_SAB_Temp_PT1&ParaNames%5B%5D=DBNS_ ...
    INFO:shiftcheck:. 244_VME_Crate_Temperature___Last_30_Minutes.png               http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=FanTemperature&ParaNames%5B%5D=Temperat ...
    INFO:__main__:wrote annotated target page to /private/tmp/4hr/annotated.html 

    real    2m1.255s
    user    0m5.214s
    sys     0m1.989s

"""
from __future__ import with_statement
import os, logging
from pprint import pformat
from lxml import etree
from lxml.etree import tostring
from urlparse import urlparse
from StringIO import StringIO
log = logging.getLogger(__name__)

def sanitize(txt):
    """
    :param txt:
    :return: txt with some characters removed and spaces swapped to '_'
    """
    txt = ' '.join(txt.split())
    txt = txt.replace(',','').replace('#','').replace(' ','_').replace(':','_').replace('&','_').replace('(','_').replace('~','-').replace(')','_')
    return txt


class Visitor(dict):
    def __init__(self, tree, aprefix=None,pull=range(10)):
        """
        :param tree:  lxml root node of tree to walk 
        :param aprefix:
        :param pull:
        """
        dict.__init__(self)
        self.tree = tree
        self.aprefix = aprefix
        self.pull = pull
        self.ctx = None
        self.count = 0
        self._visit_walk_(tree)

    def _visit_walk_( self, node ):
        """
        Recursive walk visiting every node
        :param node:
        """
        self._visit_node( node )
        for c in node:
            self._visit_walk_(c)

    def _ctx_from_link_label(self, text):
        """
        Some links suffer from not being within an ordinary li and colon context,
        determine a context for these based on the text of the link

        """
        uctx = ""
        if text[0:4] in ("DBNS","LANS","FARS"):
            uctx = text[0:4]
        if text[0:3] in ("EH1","EH2","EH3"):
            uctx = text[0:3]
        return uctx

    def _visit_node(self, node):
        """
        :param node: 

        Called for all nodes in the tree 
       
        #. for `li` nodes with text containing ":" take the text as `ctx`
        #. for selected `a` nodes which have href starting with `self.aprefix` 
           add metadata annotation attributes to the tree

        IDEA: could eliminate context extraction and just use the index, 
        making the code much less fragile : at the expense of less meaningful filenames

        """
        text = node.text.lstrip() if node.text else None
        if node.tag == 'li':
            if text.find(":")>-1:
                ctx = text.replace(":","")
            else:
                ctx = None
            self.ctx = ctx
        elif node.tag == 'a':
            href = node.attrib.get('href',None)
            if href and href.startswith(self.aprefix):
                uctx = self.ctx
                if not uctx: 
                    uctx = self._ctx_from_link_label(text) 
                self.count += 1    
                metadata = dict(index=self.count,ctx=uctx)
                node.attrib['metadata']=repr(metadata)
        else:
            pass


    def retrieve_node_(self, br, node):
        """
        :param br:
        :param node:
        """
        metadata = self.get_metadata(node)
        if not metadata:
            return

        index = metadata['index']
        url = node.attrib['href']
        name = sanitize("%(index)0.3d_%(ctx)s_%(text)s.png" % metadata) 

        if index not in self.pull:
            return

        ret = self.imgsrc_retrieve( br, url, name, nimg=-1)
        mark = "." if ret else "E"
        self[index] = mark 

        log.info("%s %-60s  %s ..." % (mark, name, url[0:100] ))
        node.text = "[%(index)0.3d] %(text)s" % metadata  # caution tree diddling, providing the link index in labels 


    def get_metadata(self, node):
        """
        Decode the metadata attribute string into a dict 
        """
        metadata_s = node.attrib.get('metadata',None)
        if not metadata_s:
            return None
        metadata = eval(metadata_s) 
        metadata['text'] = node.text
        return metadata

    def retrieve_walk_( self, br, node ):
        """
        Recursive walk visiting every node

        :param node:
        """
        self.retrieve_node_( br, node )
        for child in node:
            self.retrieve_walk_(br, child)

    def retrieve_(self, br):
        """
        """
        self.retrieve_walk_(br, self.tree)

    def imgsrc_retrieve(self, br, url, filename, nimg=-1 ):
        """
        :param br: mech.Browser wrapper class instance holding an authorized mechanize.Browser instance
        :param url:
        :param filename:
        :param nimg: index of the image on an itermediary html page
        """

        u = urlparse(url)
        udir = "%s://%s%s" % ( u.scheme, u.hostname, os.path.dirname(u.path) ) 
        log.debug("filename %s udir %s " % ( filename, udir ))

        if not br:return None

        subtree = br.open_(url, parse=True)
        imgs = subtree.xpath('.//img')
        src = imgs[nimg].attrib['src']   
        usrc = src if src.startswith('http') else "%s%s" % (udir, src)   # not nice, seems brittle : how better to absolutize potential relative link
        return br.retrieve( usrc, filename=filename )

    def write_tree(self, path):
        with open(path,"w") as fp:
            fp.write(tostring(self.tree))



if __name__ == '__main__':

    from cnf import cnf_
    from autobrowser import AutoBrowser
    cnf = cnf_(__doc__)

    br = AutoBrowser(cnf)
    targets = cnf.get('targets',"").split()
    for target in targets:
        outd = br.chdir_(target)
        tree = br.open_(target, parse=True)

        pull=range(1,cnf.npull+1)  
        v = Visitor(tree,aprefix=cnf['visitor_aprefix'],pull=pull )
        v.retrieve_( br )
        log.info("STAT\n%s\n" % pformat(v))
        path = os.path.abspath("annotated.html")
        v.write_tree(path)
        log.info("wrote annotated target page to %s " % path )

