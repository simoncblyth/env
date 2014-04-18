#!/opt/local/Library/Frameworks/Python.framework/Versions/2.6/Resources/Python.app/Contents/MacOS/Python
"""
Adapted from the docutils rst2s5.py tool 

A minimal front end to the Docutils Publisher, producing HTML slides using
the S5 template system.

* http://docutils.sourceforge.net/docs/howto/rst-roles.html

"""
try:
    import locale
    locale.setlocale(locale.LC_ALL, '')
except:
    pass

import logging, time
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import docutils.writers.s5_html as s5
from docutils.parsers.rst import Directive, directives
from docutils import nodes

class MonkeyWriter(s5.Writer):
    pass

class MonkeyS5HTMLTranslator(s5.S5HTMLTranslator):
    s5_stylesheet_template = """\
<!-- configuration parameters -->
<meta name="defaultView" content="%(view_mode)s" />
<meta name="controlVis" content="%(control_visibility)s" />
<!-- style sheet links -->
<script src="%(path)s/slides.js#Q#" type="text/javascript"></script>
<link rel="stylesheet" href="%(path)s/slides.css#Q#"
      type="text/css" media="projection" id="slideProj" />
<link rel="stylesheet" href="%(path)s/outline.css#Q#"
      type="text/css" media="screen" id="outlineStyle" />
<link rel="stylesheet" href="%(path)s/print.css#Q#"
      type="text/css" media="print" id="slidePrint" />
<link rel="stylesheet" href="%(path)s/opera.css#Q#"
      type="text/css" media="projection" id="operaFix" />\n""".replace("#Q#","?monkeykillcache=%s" % time.time() )
    pass


s5.Writer = MonkeyWriter
s5.S5HTMLTranslator = MonkeyS5HTMLTranslator


def id_mangle(title):
    """
    Cannot find where docutils comes up with the id,  there is more to it 
    than this... removing non word characters.

    Found it nodes.make_id 
    """  
    return title.lower().replace(" ","-")


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
   

directives.register_directive('s5_background_image',S5BackgroundImage)




from docutils.core import publish_cmdline, default_description
description = ('Generates S5 (X)HTML slideshow documents from standalone '
               'reStructuredText sources.  ' + default_description)
publish_cmdline(writer_name='s5', description=description)


