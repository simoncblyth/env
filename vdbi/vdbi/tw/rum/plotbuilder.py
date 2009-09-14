
from vdbi import debug_here

from tw.jquery import JQPlotWidget
from tw.jquery.jqplot import AsynchronousJQPlotWidget
from tw.api import JSLink, js_function,  js_callback





class DbiJQPlotWidget(JQPlotWidget):
    def update_params(self, d):
        super(JQPlotWidget, self).update_params(d)
        if not d.get('data'):
            raise ValueError, "JQPlotWidget must have data to graph"
        if not d.get('id'):
            d['id'] = 'jqplot_%s' % str(int(random() * 999))
        data = d.get('data', [])
        options = d.get('options', {})
        self.add_call(
            js_function('$.jqplot')( "%s" % d.id, data, options)
        )
        
        return d



class DbiAsynchronousJQPlotWidget(AsynchronousJQPlotWidget):
    """
        Unfortunately have to duplicate much of tw.jquery.jqplot code 
        as need to flatten the nested js_call to get it working with the tw tip 
        the patch in the ticket is too much to stomach 
          
        Avoid nested js_function quoting issues 
               http://toscawidgets.org/trac/tw.jquery/changeset/105%3A5ac248d9c07a
               http://toscawidgets.org/trac/tw/ticket/30
         
        Nested calls lead to incorrect quoting :        
           self.add_call(
               js_function('%s =' % d.id)(
                   js_function('$.jqplot')(
                       "%s" % d.id, data, options)))
                   
    """
 
    def __init__(self, id,  *args, **kw):
        if not(kw):kw = {}
        if 'extra_js' not in kw:
            kw['extra_js'] = [JSLink(modname = 'vdbi.tw.rum', filename = 'static/vdbi_plotter.js' )]
        super(DbiAsynchronousJQPlotWidget, self).__init__(id, *args, **kw)


    def update_params(self, d):
        """
              NB skipping AsynchronousJQPlotWidget.update_params via :
                   super(AsynchronousJQPlotWidget).update_params
        """
        super(AsynchronousJQPlotWidget, self).update_params(d)
        if not d.get('id'):
            raise ValueError, "DbiAsynchronousJQPlotWidget must have a plot id"
        if not d.get('src_url'):
            raise ValueError, "DbiAsynchronousJQPlotWidget must have a data url"
        
        src_url = d.get('src_url')
        url_kw = d.get('url_kw')
        if not url_kw:
            url_kw = {}
        data = d.get('data')
        if not data:
            data = [[[0,0]]]
        options = d.get('options')
        if not options:
            options = { 'title':"Default Title from Python" }
        interval = d.get('interval')
        if not interval:
            interval = -1
        
        self.add_call(
             js_function('Plotter.setup')( {'data_url':src_url , 'data_kw':url_kw , 'plot_id':d.id , 'opts':options }  )
            )



      
from rum import app
class DbiPlotView(DbiAsynchronousJQPlotWidget):

    def __init__(self):
        super(DbiPlotView, self).__init__("jqplotLabel")

    def data_url(self, req ):
        url = req.host_url + req.path_info + ".json?" + req.query_string   ## http://pythonpaste.org/webob/reference.html
        return url

    def update_params(self, d):
        d['id'] = "plotid"
        d['src_url'] = self.data_url( app.request )
        d = super(DbiPlotView,self).update_params(d)
        print "DbiPlotView.update_params %s " % repr(d)
        return d

