#!/usr/bin/env python
"""

* reproduces issue :env:`ticket:332`
* http://supervisord.org/logging.html

"""
import os
from bitten.build.shtools import exec_

class MockCtx(object):
    basedir = "/tmp/env/base"
    def resolve(self, p):
        return p
    def log(self, smth):
        print smth

if __name__ == '__main__':
    ctx = MockCtx()
    if not os.path.isdir(ctx.basedir):
        os.makedirs(ctx.basedir)
    kwa =  {'output': '/dev/stdout', 'executable': '/bin/bash', 'args': ' -c "   export BUILD_SLUG=dybinst/20542_20386 ; export BUILD_REVISION=20386 ;   unset SITEROOT ; unset CMTPROJECTPATH ; unset CMTPATH ; unset CMTEXTRATAGS ; unset CMTCONFIG ;   source ~/.bitten-slave/local.bash ; echo  " '}
    #del kwa['output']  omitting the output avoids the issue
    exec_(ctx, **kwa )


