
import logging
from tw.rum.repeater import JSRepeater
from tw.api import JSLink
from tw.dojo import dojo_js

log = logging.getLogger(__name__)


modname = 'vdbi.tw.rum'
dbi_repeater_js = JSLink(
   modname = modname,
   filename = 'static/vdbi_repeater.js',
   javascript=[dojo_js],
)

class DbiJSRepeater(JSRepeater):
    """
        Required to change the repeater to get DateTime pickers to work ... via
        adding calendar setup on creating repetitions 
    """
    template = "genshi:vdbi.tw.rum.templates.repeater"
    javascript = [dbi_repeater_js]
    def update_params(self, d):
        print "DbiJSRepeater %s" % repr(d)
        super(DbiJSRepeater, self).update_params(d)