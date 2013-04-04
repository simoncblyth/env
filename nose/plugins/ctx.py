"""This plugin will report the context of a test run.
To turn it on, use the ``--with-ctx`` option
or set the NOSE_WITH_CTX environment variable. 

Possibilites, stop the test run if not a clean revision

"""

import logging
import os
import sys
from datetime import datetime
from nose.plugins.base import Plugin

log = logging.getLogger('nose.plugins')

tfmt_ = lambda dt:dt.strftime("%Y-%m-%d %H:%M:%S")


class _ctx(dict):
    """
    Capturing the context of a nosetest run 

    #. userid, hostname, commandline 

    """
    tmpl = r"""
%(relpath)s @ %(version)s  %(startf)s
"""

    def __init__(self, vcmd):
        dict.__init__(self)
        self['vcmd'] = vcmd
        self['version'] = os.popen(vcmd).read().strip()
        self['start'] = datetime.now()
        self['startf'] = tfmt_(self['start'])
        self['abspath'] = os.path.abspath(os.path.curdir)
        self['siteroot'] = os.environ['SITEROOT']
        self['relpath'] = self['abspath'][len(self['siteroot'])+1:]
    __str__ = lambda _:_.tmpl % _


class Ctx(Plugin):
    """
    Use this plugin to run report the context of a test run
    """
    def options(self, parser, env):
        """Register commandline options.
        """
        Plugin.options(self, parser, env)
        parser.add_option('--ctx-vcmd', action='store', dest='ctx_vcmd',
                          default=env.get('NOSE_CTX_VCMD', 'svnversion'),
                          metavar="VCMD",
                          help="Command to determine working copy version")

    def begin(self):
        """Report context of the test run prior to start
        """
        self.ctx = _ctx(self.vcmd)
        log.debug(str(self.ctx))

    def configure(self, options, conf):
        """Configure plugin.
        """
        Plugin.configure(self, options, conf)
        self.conf = conf
        self.vcmd = options.ctx_vcmd

    def report(self, stream):
        """Output ctx report.
        """
        log.debug('printing ctx report')
        stream.write(str(self.ctx))



