

from tw.api import lazystring as _
from tw.rum.contextlinks import ContextLinks
from vdbi.dbg import debug_here
from rum import app

gen_links = ContextLinks.gen_links.im_func

    
def dbi_related_resources(resource, dbi_or_vld=False):
    """
       for any resource in the triplet : eg SimPmtSpecDbi / SimPmtSpecVld / SimPmtSpec return the other two 
    """
    name = resource.__name__
    if name.endswith('Dbi') or name.endswith('Vld'):name = name[0:-3]
    for r in app.resources.keys():
        if r.__name__.startswith(name) and r != resource:
            if dbi_or_vld:
                if r.__name__[-3:] in ('Dbi','Vld'):
                    yield r
            else:
                yield r


class DbiContextLinks(ContextLinks):
    """
          Overrides to 
             * omit the new_link_tuple
             * add related Dbi links allowing removal from layout.html resource list 
    """
    def _related_resource_name(self, resource):
         r_names, r_namep = app.names_for_resource(resource)
         return  app.translator.ugettext(r_names)
    
    def related_link_tuple(self, resource):
        routes=app.request.routes
        name = self._related_resource_name(resource)
        new_name=name[:1].upper()+name[1:]
        return (name, _(new_name), app.url_for(resource=resource, _memory=False))
    
    @gen_links.when("routes['resource'] is not None", prio=10 )
    def _gen_links_res(self, routes):
        return [self.index_link_tuple()]  + [self.related_link_tuple(r) for r in dbi_related_resources(routes['resource'],dbi_or_vld=True)] 
        
    @gen_links.when("routes['resource'] is not None and routes.get('id',None) is not None", prio=10)
    def _gen_links_obj(self, routes):
        return [  self.show_link_tuple(),
                  self.index_link_tuple(),
                ]
        

                

        
        





