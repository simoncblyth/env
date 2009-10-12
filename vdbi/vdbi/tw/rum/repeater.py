
from vdbi.dbg import debug_here

import logging
from tw.rum.repeater import JSRepeater
from tw.api import JSLink, js_function 
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
        #super(DbiJSRepeater, self).update_params(d)
        super(JSRepeater, self).update_params(d)
        log.debug("JSRepeater handling %r", d.value)
        first_id = self.children[0].id
        first_name = self.children[0].name
        
        if first_name.find("xtr") > -1 or first_name.find('plt') > -1:
            print "DbiJSRepeater enforcing a blank start for %s " % first_name
            d.repetitions = 0     ## enforce blank start 
        
        #TODO: WidgetRepeater should update d.repetitions based on extra
        d.repetitions = max(d.repetitions, len(d.value) + d.extra)
        # remove trailing "-0" 
        d.add_link_id = first_id[:-2] + '_add_trigger'
        js_args = dict(
            add_link_id = d.add_link_id,
            first_id = first_id,
            first_name = first_name,
            max_repetitions = 1000,
            max_error_text = unicode(d.max_error_text),
            error_class = d.error_class,
            )
        #debug_here()
        if d.repetitions == 0:
            d.repetitions = 1
            js_args.update(clear_on_init=True)
        d.jscall = "dojo.addOnLoad(function() {"+\
            unicode(\
                js_function(\
                    "new JSRepeater")(\
                    js_args)) +"}); "
        #debug_here()