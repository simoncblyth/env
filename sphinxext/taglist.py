
from docutils import nodes
from docutils.parsers.rst import Directive, directives

from sphinx import addnodes
from sphinx.util import parselinenos

from pprint import pformat 


class TagList(Directive):
    """
    How to access the context of where the directive appears ?
    """
    has_content = True
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {
        'linenos': directives.flag,
    }

    def run(self):
        """

        The meta is not not document specific::


	     

        """ 
        env = self.state.document.settings.env
        content = u'\n'.join(self.content)
        meta = env.metadata 

        txt = pformat(meta)
        #for tag in content.split(","):
        #    txt += "%s\n" % tag

        literal = nodes.literal_block(txt, txt)
        literal['linenos'] = False
        return [literal]


def setup(app):
    app.add_directive('taglist', TagList)



