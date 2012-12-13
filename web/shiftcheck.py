#!/usr/bin/env python

from lxml import etree
from lxml.etree import tostring
from StringIO import StringIO

def parse_( content=None, path=None ):
    """
    Perform lxml etree HTML parse on html file and return etree root instance 

    :param content: string html content
    :param path: to html file
    :return: lxml etree root instance
    """
    if not content:
        content = open(path).read()
    root = etree.parse( StringIO(content), etree.HTMLParser() ).getroot()
    return root


def sanitize(txt):
    txt = ' '.join(txt.split())
    txt = txt.replace(',','').replace('#','').replace(' ','_').replace(':','_').replace('&','_').replace('(','_').replace('~','-').replace(')','_')
    return txt

class Item(dict):
    """

    http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php?ParaNames%5B%5D=DBNS_PTH_T1&ParaNames%5B%5D=DBNS_PTH_T2&ParaNames%5B%5D=DBNS_PTH_T3&TimeSpan=1440&Interval=1&TableName=DBNS_ENV_PTH&Site=DBNS&MainSys=ENV&SubSys=PTH&ViewType=RealtimeChart
    http://dcs2.dyb.ihep.ac.cn/RealtimeImage.php?ParaNamesString=DBNS_PTH_T1%60%2C%60DBNS_PTH_T2%60%2C%60DBNS_PTH_T3&TableName=DBNS_ENV_PTH&TimeSpan=86400&Interval=1&All=

    """ 

    def __init__(self, **kwa ):
        dict.__init__(self, **kwa )
        self.name = self._name()
    def _name(self):
        ctx = sanitize(self['ctx'])
        txt = sanitize(self['text'])
        return "%s_%s" % (ctx,txt)
    #name = property(_name)


class Visitor(list):
    def __init__(self, content=None, path=None , aprefix=None):
        tree = parse_(content=content,path=path)
        #print tostring(tree)
        self.ctx = None
        self.aprefix = aprefix
        self.names = []
        self.walk_(tree)
        self.de_dupe()

    def walk_( self, node ):
        """
        """
        text = node.text.lstrip() if node.text else None
        #print node, node.tag, node.text
        if node.tag == 'li':
            if text.find(":")>-1:
                ctx = text.replace(":","")
            elif text[0:4] in ("DBNS","LANS","FARS"):
                ctx = text[0:4]
            else:
                try:
                    pass
                    #print "%s [%s] [%s]" % ( node.tag, text[0:4], text )
                except UnicodeError:
                    pass
                ctx = None
            self.ctx = ctx
        elif node.tag == 'a':
            href = node.attrib.get('href',None)
            if href and href.startswith(self.aprefix):
                uctx = self.ctx
                if not uctx: 
                    if text[0:4] in ("DBNS","LANS","FARS"):
                        uctx = text[0:4]
                    if text[0:3] in ("EH1","EH2","EH3"):
                        uctx = text[0:3]
                #print uctx, text
                item = Item(ctx=uctx,text=text,href=href)
                self.append(item)
        else:
            pass
        for c in node:
            self.walk_(c)

    def de_dupe(self):
        for n,item in enumerate(self):
            dupes = filter(lambda _:_.name == item.name, self )
            if len(dupes) > 1:
                print "de_dupe  %-3s %-40s %-40s  %s  " % ( n, item['ctx'],item['text'], item.name )
                for n,dupe in enumerate(dupes):
                    dupe.name = "%s_%s" % (dupe.name,n) 



if __name__ == '__main__':

    v = Visitor(path="ShiftCheck.html",aprefix='http://dcs2.dyb.ihep.ac.cn/RealtimeChart.php')
    print "items %s " % len(v)
    for n,item in enumerate(v):
        print "%-4s %-60s %s " % ( n, item.name, item['href'][0:100])



