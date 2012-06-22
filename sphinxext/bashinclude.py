import os

from docutils.parsers.rst import Directive, directives
from docutils.parsers.rst.directives.misc import Include as BaseInclude
from bashrst import bashrst

class BashInclude(BaseInclude):
    """
    Like the standard "Include" directive, but extracts the 
    usage message string from a collection of bash functions  
    """
    def run(self):
       	env = self.state.document.settings.env
	rel_filename, filename = env.relfn2path(self.arguments[0])
	self.arguments[0] = bashrst(filename, os.getcwd() )
	return BaseInclude.run(self)


def setup(app):
    app.add_directive('bashinclude', BashInclude)


