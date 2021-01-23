#!/usr/bin/env python

from docutils.parsers.rst import Directive, directives
from docutils import nodes

# kludge global as attempting to follow s5_video pattern scrambles the html, slides become sections 
urls = []  

class s5backgroundimage(nodes.General, nodes.Inline, nodes.Element):
    pass

class div_background(object):

    div_tmpl = r"""div.slide#%(tid)s{
             background-image: %(image)s;
             background-size: %(size)s;
             background-position: %(position)s;
             %(extra)s
          }"""

    def parse_spec(self, spec_line):
        spec_elem = spec_line.split()
        nelem = len(spec_elem)
        size, position, extra = "auto_auto", "0px_0px", ""
        if nelem > 0:url = spec_elem[0] 
        if nelem > 1:size = spec_elem[1]
        if nelem > 2:position = spec_elem[2]
        if nelem > 3:extra = spec_elem[3]
        if len(extra) > 0: 
            extra = "%s ; " % extra  
        pass
        _ = lambda _:_.replace("_"," ")
        return dict(url=url, size=_(size),position=_(position), extra=_(extra)) 

    def __init__(self, lines):
        title, specs = lines[0], lines[1:]
        self.lines = lines 
        self.title = title
        self.specs = specs
        dd = [] 
        for spec in specs:
            d = self.parse_spec(spec)
            urls.append(d["url"]) 
            dd.append(d)
        pass
        dc = {}
        dc["image"] = ",".join(map(lambda d:"url(%s)" % d["url"], dd))   
        dc["size"] = ",".join(map(lambda d:"%s" % d["size"], dd))
        dc["position"] = ",".join(map(lambda d:"%s" % d["position"], dd))
        dc["extra"] = " ".join(map(lambda d:"%s" % d["extra"], dd))
        dc["tid"] = nodes.make_id(title)

        self.html = self.div_tmpl % dc

    def __repr__(self):
        return "div_background %d" % (len(self.lines))
    def __str__(self):
        return self.html


def render_s5backgroundimage_1( n ):
    """
    Generalize to handle multiple urls, size and position 
    """
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

    content = "\n".join(filter(lambda _:len(_) == 0 or _[0] != '#', n.content)) + "\n" # remove comments and make into big string
    divs = []
    for i, item in enumerate(content.split("\n\n")):  # split on empty lines
        lines = item.split("\n")
        if len(lines) < 2: 
            print("skip single line item [%s] " % lines[0])
        else:
            div = div_background(lines)
            print(div)
            divs.append(div.html)
        pass
    pass
    html = style_tmpl % dict(divs="\n          ".join(divs))
    return html



def render_s5backgroundimage_0( n ):
    """
    Old way handles only one url per item 

    http://www.w3schools.com/cssref/tryit.asp?filename=trycss3_background-size


    Example of a one content pair in s5_background_image directive::

          Where Next for Opticks ?
          /env/presentation/nvidia/Introducing_OptiX_7.png 640px_360px 670px_300px

    """

    div_tmpl = r"""div.slide#%(tid)s{
             background-image: url(%(url)s);
             background-size: %(size)s;
             background-position: %(position)s;
             %(extra)s
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
        size, position, extra = "auto_auto", "0px_0px", ""
        if nelem > 0:url = spec_elem[0] 
        if nelem > 1:size = spec_elem[1]
        if nelem > 2:position = spec_elem[2]
        if nelem > 3:extra = spec_elem[3]
        pass
        _ = lambda _:_.replace("_"," ")
        urls.append(url) 
        if len(extra) > 0: 
            extra = "%s ; " % extra  
        pass
        divs.append( div_tmpl % dict(tid=nodes.make_id(title), url=url, size=_(size),position=_(position), extra=_(extra))) 
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

        #render = render_s5backgroundimage_0
        render = render_s5backgroundimage_1

        raw = nodes.raw('', render(n), format = 'html')

        raw.document = self.state.document
        return [raw]
 
