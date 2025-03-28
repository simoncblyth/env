
from vdbi.dbg import debug_here
from vdbi import dbi_default_plot
from rum import app
from rum.controller import ControllerFactory, CRUDController, process_output  #, resource_action, N_

from vdbi.dbg import debug_here
from vdbi.rum.param import limit as limit_, offset as offset_

from formencode.validators import Int

from rum.wsgiutil import HTTPBadRequest
from rum.router import resource_action
from rum import exceptions, app, fields, util, _, N_
from rum.controller import formats

from rum.controller import handle_exception

import logging
log = logging.getLogger(__name__)


def is_data_request( req ):
    if req.user_agent[0:4] in ['Pyth','curl']:
        return True
    return False

class DbiCRUDController(CRUDController):
    
    
#    @property
#    def input_actions(self):
#        input_methods = ('DELETE', 'POST', 'PUT', 'GET',)   ## add GET for index validation 
#        return [action for action,method in self.routeable_actions.iteritems()
#                if method.upper() in input_methods]
#    
#    def __init__(self, error_handlers=None):
#        super(DbiCRUDController, self).__init__(error_handlers)
#        self.error_handlers.update( { 'index':'index'} )
    
    @handle_exception.when((exceptions.Invalid,))
    def _handle_validation_errors(self, e, routes):
        self.validation_errors = e
        self.response.status_int = 400
        self.flash(_(u"Form has errors. Please correct"), status="alert")
        log.debug("Validation failed: %s", e)
        form_action = getattr(self, 'form_action', 'index')
        debug_here()
        return self.forward(form_action)
    
    
    N_('index')
    @formats('html', 'json', 'csv')
    @resource_action('collection', 'GET')
    def index(self, resource):
        query = self.repository.make_query(self.request.GET)
        query = self.app.policy.filter(resource, query)
        #debug_here()
        print "index: %s " % self.request.user_agent
        if query:
            if self.get_format(self.routes) not in ('csv','json'):
                ## try to keep Safari happy 
                self.response.headers.add('Content-Script-Type','text/javascript')
                if query.limit is None:
                    
                    query = query.clone(
                        limit=self.default_limit(resource)
                        #
                        )
                elif query.limit > self.app.config.get('max_page_size', 100):
                    raise HTTPBadRequest(
                        _(u"Too many results per page requested")
                        ).exception
            else:
                limit = offset = None
                if is_data_request(self.request):
                    print "vdbi.rum.controller : data feeder user_agent detected   %s " %  self.request.user_agent
                    limit = query.limit
                    offset = query.offset
                    print "vdbi.rum.controller limit %s offset %s " % ( limit, offset )
                else:
                    plotparam = query.plotparam()
                    offset = offset_(plotparam.get('offset',None))
                    limit  = limit_(plotparam.get('limit',None))
                if limit or offset:
                    query = query.clone( limit=limit, offset=offset )
                    log.debug("applying plot limit/offset to query %s" % `query` )
                #debug_here()
                
        items = self.repository.select(query)
        return {
            'items': items,
            'query': query,
            }

    
    @process_output.when("isinstance(output,dict) and self.get_format(routes) == 'json' and is_data_request(self.request)",prio=10)
    def _process_dict_as_json(self, output, routes):
        json_output = self.app.jsonencoder.encode(output)
        self.response.body = json_output
    
    @process_output.when("isinstance(output,dict) and self.get_format(routes) == 'json'", prio=10)
    def _process_dict_as_json(self, output, routes):
        q = output['query']                         
        plotseries = q.plotseries()
        plotdata = []
        for i in range(len(plotseries)):plotdata.append([])
        for i,sd in enumerate(plotseries):
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
        #print "register_crud_controller for %s" % resource
        ControllerFactory.register(DbiCRUDController, resource )




