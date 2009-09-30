import urllib
from lxml import etree
from cStringIO import StringIO   
import os

def remote_hgwebdir_repos( url = "http://belle7.nuu.edu.tw/hg" ):

    base = os.path.basename( url )
    html = urllib.urlopen(url).read()
    tree = etree.parse( StringIO(html) , etree.HTMLParser() )
    root = tree.getroot()
    repos = root.xpath('.//table/tr/td[1]/a/@href')  
    for repo in repos:
        print "%s%s" % ( base, repo )


if __name__=='__main__':
    import sys
    remote_hgwebdir_repos( sys.argv[1] )

