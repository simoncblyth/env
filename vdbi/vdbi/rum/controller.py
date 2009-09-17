
from vdbi import debug_here
from rum import app
from rum.controller import ControllerFactory, CRUDController, process_output

class DbiCRUDController(CRUDController):
    @process_output.when("isinstance(output,dict) and self.get_format(routes) == 'json'", prio=10)
    def _process_dict_as_json(self, output, routes):
        print "vdbi.rum.controller:process_output %s " % repr(output)  
        json_output = self.app.jsonencoder.encode(output)
        self.response.body = json_output
        print "json_output %s " % repr(json_output)
        debug_here()


for resource in app.resources.keys():
    ControllerFactory.register(DbiCRUDController, resource )




