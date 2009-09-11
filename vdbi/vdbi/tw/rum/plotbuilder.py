

from tw.jquery import JQPlotWidget
from tw.api import js_function

class DbiJQPlotWidget(JQPlotWidget):
    def update_params(self, d):
        super(JQPlotWidget, self).update_params(d)
        if not d.get('data'):
            raise ValueError, "JQPlotWidget must have data to graph"
        if not d.get('id'):
            d['id'] = 'jqplot_%s' % str(int(random() * 999))
        data = d.get('data', [])
        options = d.get('options', {})
        
        #  avoid nested js_function quoting issues 
        #      http://toscawidgets.org/trac/tw.jquery/changeset/105%3A5ac248d9c07a
        #      http://toscawidgets.org/trac/tw/ticket/30
        #self.add_call(
        #    js_function('%s =' % d.id)(
        #        js_function('$.jqplot')(
        #            "%s" % d.id, data, options)))
        
        self.add_call(
            js_function('$.jqplot')( "%s" % d.id, data, options)
        )
        
        return d
    

class DbiPlotView(DbiJQPlotWidget):
    data = [[[1,1],[2,2]]]
    
    def __init__(self):
        super(DbiPlotView, self).__init__("jqplotLabel")

    def update_params(self, d):
        d['id'] = "plotid"
        d = super(DbiPlotView,self).update_params(d)
        print "DbiPlotView.update_params %s " % repr(d)
        return d

