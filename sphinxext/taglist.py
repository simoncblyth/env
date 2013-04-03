import os.path
import logging
log = logging.getLogger(__name__)
from docutils import nodes
from docutils.parsers.rst import Directive, directives

from sphinx import addnodes
from sphinx.util import parselinenos

from pprint import pformat 

textnode = lambda txt:nodes.Text(txt,txt)

class docmeta_node(nodes.sidebar, nodes.Element):pass

class DocMeta(Directive):
    """
    Present document level metadata  

    Suspect that use from rst_prolog stops the metadata field definition 
    that has to be at the head of the page. 
    Works from rst_epilog however.

    TODO: 
    
    #. generate pages per tag that list the tagged docs OR link to such docs if they exist  
    #. make invisible when no metadata

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
        problem is that the doc metadata is not available at directive run time it seems
        """
        env = self.state.document.settings.env
        #log.info("DocMeta.run  docname %s " % (env.docname))
        dmn = docmeta_node()
        dmn.docname = env.docname
        return [dmn]

def process_docmeta(app, doctree):
    env = app.builder.env
    #log.info("process_docmeta START")
    for node in doctree.traverse(docmeta_node):
        docname = getattr(node, 'docname')
        docmeta = env.metadata.get(docname)
        #log.info("process_docmeta docname %s docmeta %s " % ( docname, docmeta ))
        para = nodes.paragraph()
        for k,v in docmeta.items():
            para += textnode("%s : %s " % (k,v) )
        node += para 
    pass    

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




def visit_docmeta_node(self, node):
    self.visit_sidebar(node)
def depart_docmeta_node(self, node):
    self.depart_sidebar(node)




def setup(app):
    app.add_directive('taglist', TagList)
    app.add_directive('docmeta', DocMeta)

    app.add_node( docmeta_node, html=(visit_docmeta_node, depart_docmeta_node ))


    app.connect('doctree-read', process_docmeta)

