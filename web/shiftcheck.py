#!/usr/bin/env python
"""
Usage::

   ./shiftcheck.py

Parses the ShiftCheck html, extracting and naming all links corresponsing to DCS monitoring pages


"""
from __future__ import with_statement
import os, logging
from pprint import pformat
from lxml import etree
from lxml.etree import tostring
from urlparse import urlparse
from StringIO import StringIO
log = logging.getLogger(__name__)

def parse_( content=None, path=None ):
    """
    Perform lxml etree HTML parse on html string or path to a file
    and return etree root instance 

    :param content: string html content
    :param path: to html file
    :return: lxml etree root instance
    """
    if not content:
        content = open(path).read()
    root = etree.parse( StringIO(content), etree.HTMLParser() ).getroot()
    return root

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
        self.stat = dict(ok=[],error=[])

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
        """
        metadata = self.get_metadata(node)
        if not metadata:
            return

        index = metadata['index']
        url = node.attrib['href']
        name = sanitize("%(index)0.3d_%(ctx)s_%(text)s.png" % metadata) 

        if index in self.pull:
            ret = self.imgsrc_retrieve( br, url, name, nimg=-1)
            if not ret:
                mark = "E"
                self.stat['error'].append(index)
            else:
                mark = "."
                self.stat['ok'].append(index)
        else:        
            mark = " "

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
    logging.basicConfig(level=logging.DEBUG)
    name="ShiftCheck.html"
    tree = parse_(path=name)
    v = Visitor(tree,aprefix='http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php')
    br = None
    v.retrieve_(br)
    v.write_tree("annotated_%s" % name )

    print pformat(v.stat) 


