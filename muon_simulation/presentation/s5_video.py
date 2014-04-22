#!/usr/bin/env python
"""
Following docutils techniques from 

* https://github.com/astraw/burst_s5/blob/master/burst_s5/video_directive.py


S5 slides, have a click on slide moves to next slide



"""
from docutils.parsers.rst import Directive, directives
from docutils import nodes

att_ = lambda k,v:'%(k)s="%(v)s"' % locals()
count = 0

class s5video(nodes.General, nodes.Inline, nodes.Element):
    pass

def render_s5video( n ):
    video_tmpl = r"""
        <style type="text/css">
           video.flic {
              text-align: center;
          }
        </style>
        <video id="%(id)s" class="flic" src="%(src)s" controls %(height)s %(width)s %(poster)s >
            <p> Your Browser does not support HTML5 Video </p>
        </video>
    """
    global count
    count += 1 
    
    src = n.arguments[0]
    height = n.options.get('height','480')
    width = n.options.get('width','640')
    poster = n.options.get('poster','')

    ctx = {}
    ctx['id'] = 's5video_%s' % count
    ctx['src'] = src
    ctx['height'] = att_('height',height)
    ctx['width'] = att_('width',width)
    ctx['poster'] = att_('poster',poster) if len(poster) > 0 else ''

    return self.video_tmpl % ctx

def visit_s5video_html(self, n ):
    html = render_s5video( n )
    self.body.append(html)

def depart_s5video_html(self, n ):
    pass

class S5VideoDirective(Directive):
    has_content = False
    required_arguments = 1
    optional_arguments = 1 
    final_argument_whitespace = True
    option_spec = {
       'height':str,
       'width':str,
       'poster':str,
    }
    def run(self):
        n = s5video()
        n.arguments = self.arguments
        n.options = self.options.copy()
        return [n]
             


