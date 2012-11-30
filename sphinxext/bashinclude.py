import os

from docutils.parsers.rst import Directive, directives
from docutils.parsers.rst.directives.misc import Include as BaseInclude

from env.doc.bash2rst import Bash

class BashInclude(BaseInclude):
    """
    Like the standard "Include" directive, but extracts the 
    usage message string from a collection of bash functions  
    """
    def run(self):
        env = self.state.document.settings.env
        rel_filename, filename = env.relfn2path(self.arguments[0])
        b = Bash(filename)
        gpath = b.write_rst()
        self.arguments[0] = gpath
        return BaseInclude.run(self)

def setup(app):
    app.add_directive('bashinclude', BashInclude)
