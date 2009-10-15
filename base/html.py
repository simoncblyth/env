import urllib
from lxml import etree
from cStringIO import StringIO   
import os

def html_href( url , xpath='.//a/@href' ):
    """
        Parse the html obtained from the url and emit
        urls of links found
         
        Usage :
           python ~/e/base/html.py $(firefox-url) 

    """
    base = os.path.dirname( url )
    html = urllib.urlopen(url).read()
    tree = etree.parse( StringIO(html) , etree.HTMLParser() )
    root = tree.getroot()
    hrefs = root.xpath( xpath )  
    for href in hrefs:
        if href.startswith('http'):
            print href
        else:
            print "%s%s" % ( base, href )


if __name__=='__main__':
    import sys
    html_href( sys.argv[1] )

