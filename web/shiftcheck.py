#!/usr/bin/env python
"""
Usage::

   ./shiftcheck.py

Parses the ShiftCheck html, extracting and naming all links corresponsing to DCS monitoring pages
"""
import logging
from lxml import etree
from lxml.etree import tostring
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


class Item(dict):
    """
    Holder for links
    """ 
    def __init__(self, **kwa ):
        dict.__init__(self, **kwa )
        self.name = self._name()
    def _name(self):
        ctx = sanitize(self['ctx'])
        txt = sanitize(self['text'])
        return "%s_%s" % (ctx,txt)


class Visitor(list):
    def __init__(self, tree, aprefix=None):
        """
        :param tree:  lxml root node of tree to walk 
        :param aprefix:
        """
        self.aprefix = aprefix
        self.ctx = None
        self.walk_(tree)
        self.de_dupe()

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
        #. collect href and labels of `a` nodes which have href starting with `self.aprefix` 
           into Item instances within this list 
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
                item = Item(ctx=uctx,text=text,href=href)
                self.append(item)
        else:
            pass

    def walk_( self, node ):
        """
        Recursive walk visiting every node

        :param node:
        """
        self._visit_node( node )
        for c in node:
            self.walk_(c)

    def de_dupe(self):
        """
        Look for items within this list that have duplicated names.
        Break such degeneracies with an name change to include an index.
        """
        for n,item in enumerate(self):
            dupes = filter(lambda _:_.name == item.name, self )
            if len(dupes) > 1:
                log.debug("de_dupe  %-3s %-40s %-40s  %s  " % ( n, item['ctx'],item['text'], item.name ))
                for n,dupe in enumerate(dupes):
                    dupe.name = "%s_%s" % (dupe.name,n) 


    def retrieve(self, br , npull=1):
        """
        :param br: mech.Browser wrapper class instance holding an authorized mechanize.Browser instance
        :param limit:  restrict the number of items to retreive, set to low value for testing
        """
        stat = dict(ok=[],httperror=[])
        for n,item in enumerate(self):
            if n>=npull:continue
            log.info("%0.3d %-60s %s " % ( n+1, item.name, item['href'][0:100]))
            name = "%0.3d_%s.png" % (n+1,item.name)
            url = item['href']
            log.debug("retrieve %s %s " % ( name, url ))
            tree = br.open_(url, parse=True)
            imgs = tree.xpath('.//img')
            assert len(imgs) == 3, imgs
            src = imgs[-1].attrib['src']
            usrc = "http://dcs2.dyb.ihep.ac.cn/%s" % src
            r = br.retrieve( usrc, filename=name )
            if not r:
                stat['httperror'].append(name)
            else:
                stat['ok'].append(name)
            pass    
        return stat



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tree = parse_(path="ShiftCheck.html")
    #print tostring(tree)
    v = Visitor(tree,aprefix='http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php')
    for n,item in enumerate(v):
        print "%-4s %-60s %s " % ( n, item.name, item['href'][0:100])
    print "items %s " % len(v)



