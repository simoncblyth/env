
from vdbi.dbg import debug_here
from vdbi import VLD_TIMEATTS
from vdbi import dbi_default_plot
from rum.query import Query

from rum import exceptions
from formencode.validators import Int

from tw.api import JSLink, js_function,  js_callback
from tw.api import Widget

from tw.jquery import JQPlotWidget
from tw.jquery.jqplot import AsynchronousJQPlotWidget, jqp_cursor_js, jqp_dateAxis_js 


import vdbi.rum.controller   ## use specialized process_output for json plot data 
from vdbi.rum.urls import json_url



plotter_js = JSLink(modname = 'vdbi.tw.rum', filename = 'static/vdbi_plotter.js' )

extra_js = [
   jqp_cursor_js, 
   jqp_dateAxis_js,
   plotter_js, 
]


class DbiJQPlotWidget(JQPlotWidget):
    def update_params(self, d):
        super(JQPlotWidget, self).update_params(d)
        if not d.get('data'):
            raise ValueError, "JQPlotWidget must have data to graph"
        if not d.get('id'):
            d['id'] = 'jqplot_%s' % str(int(random() * 999))
        data = d.get('data', [])
        options = d.get('options', {})
        self.add_call( js_function('$.jqplot')( "%s" % d.id, data, options))
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
            kw['extra_js'] = extra_js
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
            options = {}
        interval = d.get('interval')
        if not interval:
            interval = -1
        
        self.add_call(
             js_function('Plotter.setup')( {'data_url':src_url , 'data_kw':url_kw , 'plot_id':d.id , 'opts':options }  )
            )


     
from rum import app
from vdbi.rum.query import _vdbi_uncast






class JSONLink(Widget):
    template="genshi:vdbi.tw.rum.templates.json_link"
    def update_params(self, d):
        super(JSONLink, self).update_params(d)
        d['link']=json_url(d)


class DbiPlotView(DbiAsynchronousJQPlotWidget):

    template = "genshi:vdbi.tw.jquery.templates.jqplot"
    def __init__(self):
        super(DbiPlotView, self).__init__("jqplotLabel")

    def adapt_value_custom(self, value):
        if isinstance(value, Query):
            value = value.as_dict()
            value = _vdbi_uncast(value)
        return value

    def update_params(self, d):
        d['id'] = "plotid"
        d['src_url'] = json_url( d )
     
        #debug_here()
        plotparam = {}
        if isinstance(d['value'], Query):
            q = d['value']
            plotparam = q.plotparam()
            v = self.adapt_value_custom( q )
        else:
            v = d['value']
  
        from vdbi.rum.param import width as width_, height as height_, offset as offset_, limit as limit_
        width = width_(plotparam.get('width',None))
        height = height_(plotparam.get('height',None))                 
        offset = offset_(plotparam.get('offset',None))
        limit  = limit_(plotparam.get('limit',None))
                     
        d['width'] = '%spx' % width
        d['height'] = '%spx' % height
        
        opts = {
             'title':"Limit:%s Offset:%s " % ( limit, offset ),
             'legend':{ 'show':True }, 
             'cursor':{ 'zoom':True, 'showTooltip':False },
                'axes':{},
              'series':[],
        }
 
        xtime = True
        ytime = True
        series = []
 
        if 'q' in v and 'plt' in v['q']:       
            sdc = v['q']['plt']['c']
        else:
            routes=app.request.routes
            sdc = dbi_default_plot( routes['resource'] )
                
        for sd in sdc:
            if 'x' in sd and 'y' in sd:
                series.append( {'label':"%s:%s" % (sd['x'],sd['y']) })
                if sd['x'] not in VLD_TIMEATTS:xtime = False
                if sd['y'] not in VLD_TIMEATTS:ytime = False
                    
        if xtime:
            opts['axes']['xaxis'] = { 'renderer':'DateAxisRenderer', } 
        if ytime:
            opts['axes']['yaxis'] = { 'renderer':'DateAxisRenderer', }
                              
        if len(series) > 0:
            opts['series'] = series   
                     
        d['options'] = opts
                     
        
        super(DbiPlotView,self).update_params(d)
        #print "DbiPlotView.update_params %s " % repr(d)
        return d

