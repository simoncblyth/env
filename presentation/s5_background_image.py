#!/usr/bin/env python

import logging
log = logging.getLogger(__name__)
from docutils.parsers.rst import Directive, directives
from docutils import nodes

# kludge global as attempting to follow s5_video pattern scrambles the html, slides become sections 
urls = []  
divs = []

URLS = []


class s5backgroundimage(nodes.General, nodes.Inline, nodes.Element):
    pass

class div_background(object):

    div_tmpl = r"""div.slide#%(tid)s{
             background-image: %(image)s;
             background-size: %(size)s;
             background-position: %(position)s;
             %(extra)s
          }"""

    @classmethod
    def Find(cls, title):
        for div in divs:
            if div.ltitle == title.lower():
                return div
            pass
        return None 

    @classmethod
    def FindMeta(cls, q_meta):
        select_divs = []
        for div in divs:
            for meta in div.meta:
                if meta.find(q_meta) > -1:
                    select_divs.append(div)
                pass
            pass
        return select_divs

    def parse_spec(self, spec_line):
        spec_elem = spec_line.split()
        nelem = len(spec_elem)
        url = "parse_spec_FAILED"
        size, position, extra, meta = "auto_auto", "0px_0px", "", ""
        if nelem > 0:url = spec_elem[0] 
        if nelem > 1:size = spec_elem[1]
        if nelem > 2:position = spec_elem[2]
        if nelem > 3:extra = spec_elem[3]
        if len(extra) > 0: 
            if extra.startswith("meta:"):
                meta = extra[len("meta:"):]
            else:
                extra = "%s ; " % extra  
            pass
        pass

        global URLS
        URLS.append(url)

        if len(meta) > 0:
            log.info("%3d : spec_line %s meta %s " % (len(URLS), spec_line, meta))  
        else:
            log.debug("%3d : spec_line %s no-meta " % (len(URLS), spec_line))  
        pass
        _ = lambda _:_.replace("_"," ")
        return dict(url=url, size=_(size),position=_(position), extra=_(extra), meta=meta ) 

    def __init__(self, lines):
        title, specs = lines[0], lines[1:]
        self.lines = lines 
        self.title = title
        self.ltitle = title.lower()
        self.specs = specs
        self.urls = []
        self.meta = []
        dd = [] 
        for spec in specs:
            d = self.parse_spec(spec)
            url = d["url"]
            meta = d["meta"]
            urls.append(url)
            self.urls.append(url)  
            self.meta.append(meta)
            dd.append(d)
        pass
        dc = {}
        dc["image"] = ",".join(map(lambda d:"url(%s)" % d["url"], dd))   
        dc["size"] = ",".join(map(lambda d:"%s" % d["size"], dd))
        dc["position"] = ",".join(map(lambda d:"%s" % d["position"], dd))
        dc["extra"] = " ".join(map(lambda d:"%s" % d["extra"], dd))
        dc["tid"] = nodes.make_id(title)

        self.html = self.div_tmpl % dc
        divs.append(self)


    def __repr__(self):
        return "div_background l/s/u %d %d %d  title %s url0 %s " % (len(self.lines), len(self.specs), len(self.urls), self.title, self.urls[0] )

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
    #global urls  # kludge

    content = "\n".join(filter(lambda _:len(_) == 0 or _[0] != '#', n.content)) + "\n" # remove comments and make into big string
    divs = []
    for i, item in enumerate(content.split("\n\n")):  # split on empty lines
        lines = item.split("\n")

        #print(i)
        #print("\n".join(lines))

        if len(lines) < 2: 
            log.debug("skip single line item [%s] .. happens at tail " % lines[0])
        else:
            div = div_background(lines)
            #print(div)
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
        size, position, extra, meta = "auto_auto", "0px_0px", "", ""
        if nelem > 0:url = spec_elem[0] 
        if nelem > 1:size = spec_elem[1]
        if nelem > 2:position = spec_elem[2]
        if nelem > 3:extra = spec_elem[3]
        pass
        _ = lambda _:_.replace("_"," ")
        urls.append(url) 
        if len(extra) > 0: 
            if extra.startswith("meta:"): 
                 pass
                 meta = extra[len("meta:"):]
                 extra = ""
            else:
                 extra = "%s ; " % extra  
            pass
        pass
        divs.append( div_tmpl % dict(tid=nodes.make_id(title), url=url, size=_(size),position=_(position), extra=_(extra), meta=mera )) 
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
 
