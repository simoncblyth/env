#!/usr/bin/env python

from docutils.parsers.rst import Directive, directives
from docutils import nodes

# kludge global as attempting to follow s5_video pattern scrambles the html, slides become sections 
urls = []  

class s5backgroundimage(nodes.General, nodes.Inline, nodes.Element):
    pass

def render_s5backgroundimage( n ):
    """
    http://www.w3schools.com/cssref/tryit.asp?filename=trycss3_background-size
    """

    div_tmpl = r"""div.slide#%(tid)s{
             background-image: url(%(url)s);
             background-size: %(size)s;
             background-position: %(position)s;
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
    global urls  # kludge
    content = filter(lambda _:_[0] != '#',filter(lambda _:len(_) > 0, n.content))
    assert len(content) % 2 == 0  
    divs = []
    for pair in [content[i:i+2] for i in range(0, len(content), 2)]:
        title, spec_line = pair
        spec_elem = spec_line.split()
        nelem = len(spec_elem)
        size, position = "auto_auto", "0px_0px"
        if nelem > 0:url = spec_elem[0] 
        if nelem > 1:size = spec_elem[1]
        if nelem > 2:position = spec_elem[2]
        pass
        _ = lambda _:_.replace("_"," ")
        urls.append(url)  
        divs.append( div_tmpl % dict(tid=nodes.make_id(title), url=url, size=_(size),position=_(position))) 
    pass
    html = style_tmpl % dict(divs="\n          ".join(divs))
    return html

def visit_s5backgroundimage_html(self, n ):
    html = render_s5backgroundimage( n )
    self.body.append(html)

def depart_s5backgroundimage_html(self, n ):
    pass

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
            //localhost/env/test/LANS_AD3_CoverGas_Humidity.png 50%_auto

            JUNO_Multipurpose_Pedro_NuFact_2019
            /env/presentation/juno/JUNO_Multipurpose_Pedro_NuFact_2019.png 960px_720px 100px_0px

           
    """
    has_content = True
    required_arguments = 0
    optional_arguments = 0 
    final_argument_whitespace = False
    option_spec = { 
        'linenos': directives.flag,
    }   

    def run(self):
        n = s5backgroundimage()
        n.content = self.content

        #n.document = self.state.document
        #return [n]
        #
        # attempting to return the node (deferred rendering) rather than immediately 
        # provide raw html results in traversability of the instance 
        # but messes up the document html, 
        #    slides become sections and only 2 pages show 
        #

        raw = nodes.raw('', render_s5backgroundimage(n), format = 'html')
        raw.document = self.state.document
        return [raw]
 
