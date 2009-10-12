
from vdbi.dbg import debug_here
from vdbi import dbi_default_plot
from rum import app
from rum.controller import ControllerFactory, CRUDController, process_output  #, resource_action, N_


import IPython ; debug_here = IPython.Debugger.Tracer()




class DbiCRUDController(CRUDController):
    @process_output.when("isinstance(output,dict) and self.get_format(routes) == 'json'", prio=10)
    def _process_dict_as_json(self, output, routes):
        print "vdbi.rum.controller:process_output %s " % repr(output)

        v = output['query'].as_dict_for_widgets()
        
        print "v %s " % repr(v)
        if 'q' in v and 'plt' in v['q']:       
             sdc = v['q']['plt']['c']
        else:
             sdc = dbi_default_plot( routes['resource'] )
        
        plotdata = []
        for i in range(len(sdc)):plotdata.append([])
        for i,sd in enumerate(sdc):
            xy = lambda item:[getattr(item,sd['x']),getattr(item,sd['y'])]  
            for item in output['items']:        ##  output['items'] isa sqlalchemy.orm.query.Query   
                name = item.__class__.__name__
                plotdata[i].append( xy(item) )       
        output['plotdata'] = plotdata
        del output['items']     
        self.response.body = self.app.jsonencoder.encode(output)
        #debug_here()

#
#    N_('login')
#    @resource_action('collection', 'POST')
#    def login(self):
#        self.flash(_(u'Succesfully logged in ') )
#        self.app.redirect_to(action='index', _use_next=True, id=None)
#


def register_crud_controller():
    for resource in app.resources.keys():
        print "register_crud_controller for %s" % resource
        ControllerFactory.register(DbiCRUDController, resource )




