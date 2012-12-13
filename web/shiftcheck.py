#!/usr/bin/env python
"""
Usage::

   ./shiftcheck.py

Parses the ShiftCheck html, extracting and naming all links corresponsing to DCS monitoring pages


Indice mismatch between the PNG and the html, PNG indices::

        '200_DBNS_DBNS_pool___product_water_resistivity.png',
        '201_LANS_LANS_pool___product_water_resistivity.png',
        '202_FARS_FARS_pool___product_water_resistivity.png',
        '203_DBNS_DBNS_inflow___outflow_water_oxygen.png',
        '204_LANS_LANS_inflow___outflow_water_oxygen.png',
        '205_FARS_FARS_inflow___outflow_water_oxygen.png',
        '206_Hall_4_outflow_water_pressure.png',
        '207_Hall_4_EDI_water_resistivity.png',
        '208_Hall_4_EDI_water_oxygen.png',
        '209_Hall_4_reverse_osmosis_water__PH_conductivity_level_low.png',

And these are the annotated HTML indices::

Hall 4: 
    [200] outflow water pressure 
    [201] EDI water resistivity 
    [202] EDI water oxygen 
    [203] reverse osmosis water: PH, conductivity, level low 

EH1/EH2/EH3 (DBNS/LANS/FARS) product resistivity should be above ~(14/16/13).

    [204] DBNS pool & product water resistivity 
    [205] LANS pool & product water resistivity 
    [206] FARS pool & product water resistivity 

DBNS/LANS/FARS inflow and outflow water oxygen

    [207] DBNS inflow & outflow water oxygen 
    [208] LANS inflow & outflow water oxygen 
    [209] FARS inflow & outflow water oxygen 




"""
from __future__ import with_statement
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
    def __init__(self, tree, aprefix=None,annotate=False):
        """
        :param tree:  lxml root node of tree to walk 
        :param aprefix:
        :param annotate:
        """
        self.tree = tree
        self.aprefix = aprefix
        self.annotate = annotate
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
                index = len(self)+1    
                item = Item(index=index,ctx=uctx,text=text,href=href)
                self.append(item)
                if self.annotate:
                    node.text = "[%0.3d] %s" % ( index, node.text )
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
        for item in self:
            if item['index']>=npull:continue
            log.info("%0.3d %-60s %s " % ( item['index'], item.name, item['href'][0:100]))
            name = "%0.3d_%s.png" % (item['index'],item.name)
            url = item['href']
            log.debug("retrieve %s %s " % ( name, url ))
            subtree = br.open_(url, parse=True)
            imgs = subtree.xpath('.//img')
            assert len(imgs) == 3, imgs
            src = imgs[-1].attrib['src']
            usrc = "http://dcs2.dyb.ihep.ac.cn/%s" % src   # not nice, need better way to convert a relative to absolute URL
            r = br.retrieve( usrc, filename=name )
            if not r:
                stat['httperror'].append(name)
            else:
                stat['ok'].append(name)
            pass    
        return stat


    def write_annotated(self, path):
        if self.annotate:
            log.info("writing annotated tree to %s " % path )
            with open(path,"w") as fp:
                fp.write(tostring(self.tree))



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    name="ShiftCheck.html"
    tree = parse_(path=name)
    v = Visitor(tree,aprefix='http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php',annotate=True)
    for item in v:
        print "%0.3d %-60s %s " % ( item['index'], item.name, item['href'][0:100])
    print "items %s " % len(v)
    v.write_annotated("annotated_%s" % name )


