#!/usr/bin/env python
"""
Following docutils techniques from 

* https://github.com/astraw/burst_s5/blob/master/burst_s5/video_directive.py

S5 slides, have a click on slide moves to next slide

"""

import logging
log = logging.getLogger(__name__)
from docutils.parsers.rst import Directive, directives
from docutils import nodes

att_ = lambda k,v:'%(k)s="%(v)s"' % locals()
count = 0

class s5talk(nodes.General, nodes.Inline, nodes.Element):
    pass

def render_s5talk( n ):
    tmpl = r"""
        <!--  s5talk  -->
        <br/>
    """
    global count
    count += 1 
    
    #log.info(" count %d content %s " % (count, n.content))

    return tmpl 

def visit_s5talk_html(self, n ):
    html = render_s5talk( n )
    self.body.append(html)

def depart_s5talk_html(self, n ):
    pass

class S5TalkDirective(Directive):
    has_content = True
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = True
    options = {}

    def run(self):
        n = s5talk()
        n.content = self.content
        n.arguments = self.arguments
        n.options = self.options.copy()
        return [n]
             


