#!/usr/bin/env python

from docutils.parsers.rst import Directive, directives
from docutils import nodes

class S5BackgroundImage(Directive):
    """ 
    Usage::

        .. s5_background_image::

            Full Screen
            images/chroma/chroma_dayabay_adlid.png
         
            Full Screen 2
            images/chroma/chroma_dayabay_pool_pmts.png

            Test Server Relative Link  
            /env/test/LANS_AD3_CoverGas_Humidity.png

            Test Protocol Relative Link
            //localhost/env/test/LANS_AD3_CoverGas_Humidity.png
           
    """
    has_content = True
    required_arguments = 0
    optional_arguments = 0 
    final_argument_whitespace = False
    option_spec = { 
        'linenos': directives.flag,
    }   

    div_tmpl = r"""div.slide#%(tid)s{
             background-image: url(%(url)s);
          }"""

    style_tmpl = r"""
       <style type="text/css">

          div.slide { 
             background-clip: border-box;
             background-repeat: no-repeat;
             height: 100%%;
          }
          %(divs)s 

       </style>
    """

    def run(self):
        content = filter(lambda _:_[0] != '#',filter(lambda _:len(_) > 0, self.content))
        assert len(content) % 2 == 0  
        divs = []
        for pair in [content[i:i+2] for i in range(0, len(content), 2)]:
            title, url = pair
            divs.append( self.div_tmpl % dict(tid=nodes.make_id(title), url=url) )
        pass
        html = self.style_tmpl % dict(divs="\n          ".join(divs))
        raw = nodes.raw('', html, format = 'html')
        raw.document = self.state.document
        return [raw]
 
