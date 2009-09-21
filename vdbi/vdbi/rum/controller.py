
from vdbi import debug_here, DEFAULT_ATT_X, DEFAULT_ATT_Y
from rum import app
from rum.controller import ControllerFactory, CRUDController, process_output

class DbiCRUDController(CRUDController):
    @process_output.when("isinstance(output,dict) and self.get_format(routes) == 'json'", prio=10)
    def _process_dict_as_json(self, output, routes):
        print "vdbi.rum.controller:process_output %s " % repr(output)

        v = output['query'].as_dict_for_widgets()
        print "v %s " % repr(v)
        if 'q' in v and 'plt' in v['q']:       
             sdc = v['q']['plt']['c']
        else:
             sdc = [{'x':DEFAULT_ATT_X, 'y':DEFAULT_ATT_Y}]
        
        plotdata = []
        for i in range(len(sdc)):plotdata.append([])
        for i,sd in enumerate(sdc):
            xy = lambda item:[getattr(item,sd['x']),getattr(item,sd['y'])]  
            for item in output['items']:        ##  output['items'] isa sqlalchemy.orm.query.Query   
                plotdata[i].append( xy(item) )       
        output['plotdata'] = plotdata
        del output['items']     
        self.response.body = self.app.jsonencoder.encode(output)
        #debug_here()




for resource in app.resources.keys():
    ControllerFactory.register(DbiCRUDController, resource )




