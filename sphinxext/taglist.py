import os.path
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
        The meta is not document specific::
        """ 
        env = self.state.document.settings.env
        content = u'\n'.join(self.content)
        metadata = env.metadata 

        # add modification time to the metadata for each document
        for name,meta in metadata.items():
            meta['mtime'] = os.path.getmtime(name + '.rst')

        # dump the metadata in modification time order
        # can use the below to implement a lastest changes directive with links to those docs    
        # also add options to provide filtering based on metadata, eg listing docs with certain tags 
        #
        for i,(name,meta) in enumerate(sorted(metadata.items(), key=lambda _:_[1]['mtime'],reverse=True)):
            print i,name,meta

        txt = pformat(metadata)
        literal = nodes.literal_block(txt, txt)
        literal['linenos'] = False
        return [literal]


def setup(app):
    app.add_directive('taglist', TagList)



