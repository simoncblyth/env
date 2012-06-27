#!/usr/bin/env python
from __future__ import with_statement
import os, logging
log = logging.getLogger(__name__)

from env.doc.bash2rst import bashdoc, bashtoc


if __name__ == '__main__':
    path = os.path.expandvars("$ENV_HOME/env.bash")
    content, name = bashdoc(path)
    print "content of %s \n %s \n" % (name, content )
    bashtoc(content)

