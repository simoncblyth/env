#!/usr/bin/env python

from docutils.parsers.rst import Directive, directives
from docutils import nodes


class S5VideoRaw(Directive):
    """
    Usage::

      .. s5_video:: /env/daeview_Movie_ipad.m4v
          :height:480
          :width:640
          :poster: 


              display: box;
              box-align: center;
              box-pack: center;
              margin-left: auto;
              margin-right: auto;

              display: -webkit-box;
              -webkit-box-pack: center;
              -webkit-box-align: center;



    See also:

    * https://github.com/astraw/burst_s5
    * https://github.com/astraw/burst_s5/blob/master/burst_s5/video_directive.py

    """
    has_content = False
    required_arguments = 1
    optional_arguments = 1 
    final_argument_whitespace = True
    option_spec = {
       'height':str,
       'width':str,
       'poster':str,
    }

    video_tmpl = r"""
        <style type="text/css">
           video.flic {
          }
        </style>
        <div style="text-align: center;" >
        <video class="flic" src="%(src)s" controls height="%(height)s" width="%(width)s" %(poster)s >
            <p> Your Browser does not support HTML5 Video </p>
        </video>
        </div>
    """

    def run(self):
        src = self.arguments[0]
        height = self.options.get('height','480')
        width = self.options.get('width','640')
        poster = self.options.get('poster','')
        if len(poster) > 0:
            poster = 'poster="%s"' % poster

        html = self.video_tmpl % dict(src=src,height=height,width=width,poster=poster)
        raw = nodes.raw('', html, format = 'html')
        raw.document = self.state.document
        return [raw]
   



