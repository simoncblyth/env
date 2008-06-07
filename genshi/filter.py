#!/usr/bin/env python

import sys
import os

from genshi.builder import tag
from genshi.output import TextSerializer
from genshi.filters import Transformer
from genshi.input import HTML
from genshi import QName
from genshi.filters.transform import ENTER, EXIT, TEXT

def classify_href(href):
    if href.find("/tags/")>-1:
        c = "tag"
    elif href.find("/wiki/")>-1:
        c = "wiki"
    else:
        c = None
    return c
    
def noop(stream):
    """A filter that doesn't actually do anything with the stream."""
    for kind, data, pos in stream:
        yield kind, data, pos
        
def afilter(stream):
    """A filter that doesn't actually do anything with the stream."""
    actx = None
    for kind, data, pos in stream:
        if kind=="START":
            actx = None
            if data[0] == "a":
                href = data[1].get("href")
                actx = classify_href(href)
            #print kind, data[0]
        elif kind=="TEXT":
            #print kind, actx, data
            if actx!=None and len(data.strip())>0:
                data = "%s:%s" % ( actx, data)
                actx = None
        else:
            pass
        yield kind, data, pos

def genmarkup():
    """ generate some test markup """
    def link(resource):
        return tag.a(resource, href="http://whatever" )
    ul = tag.ul('\n',class_='taglist')
    desc = "descritpion "
    for resource in range(10):
        li = tag.li(link(resource), desc )
        ul(li, '\n')
    return ul.generate()



if '__main__'==__name__:

    if len(sys.argv)>1:
        path = sys.argv[1]
        if os.path.exists(path):
            html=HTML(file(path).read())
            print html | Transformer("li").prepend(" * ") | afilter | TextSerializer()
        else:
            print "path %s does not exist " % path
    else:
        print "expecting argument with  a path to a html file to parse and transform "



