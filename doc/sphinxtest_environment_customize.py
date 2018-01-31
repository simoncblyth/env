"""
Incorporate this into conf.py of Sphinx conf.py with::

    from env.doc.sphinxtest_environment_customize import extensions, extlinks

"""
#

__all__ = ['extensions', 'extlinks' ]


## enable logging in extensions

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s:%(lineno)-3d %(levelname)-8s %(message)s" )


## external uri warning suppresion: that works in 1.2 

import sphinx.environment
from docutils.utils import get_source_line

def _warn_node(self, msg, node):
    if not msg.startswith('nonlocal image URI found:'):
        self._warnfunc(msg, '%s:%s' % get_source_line(node))

sphinx.environment.BuildEnvironment.warn_node = _warn_node


## selection of extensions

extensions = [ 
                 'sphinx.ext.extlinks',
                 'workflow.sphinxext.wimg',
                 'workflow.sphinxext.wrel',
              ]   

## configuraing the extlinks extension

from workflow.doc.extlinks import get_extlinks
extlinks = get_extlinks()





