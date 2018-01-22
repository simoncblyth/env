#!/usr/bin/env python

import re

class EscapeURL(object):
    """
    Underscores are significant to RST, but that often occur in URLs : so need to escape em 
    """
    urlptn = re.compile('(?P<url>http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)', re.UNICODE)

    def __init__(self, ctx):
        self.ctx = ctx

    def replace(self, fullmatch):
        d = fullmatch.groupdict()
        url = d['url']
        return url.replace("_","\_")

    def __call__(self, line):
        return re.sub(self.urlptn, self.replace, line)
        

if __name__ == '__main__':
    pass

    lines = r"""

    * some non url

    * http://www.google.com
    * http://www.google.com/some_funny_text_
    * some non url
    
    """.strip().split("\n")

    eu = EscapeURL()

    print "\n".join(lines)
 
    print "~" * 100 

    print "\n".join( map(eu, lines))

