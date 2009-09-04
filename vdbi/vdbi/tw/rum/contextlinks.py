from tw.rum.contextlinks import ContextLinks

gen_links = ContextLinks.gen_links.im_func

class DbiContextLinks(ContextLinks):
    """
         Override to omit the new_link_tuple
    """
    @gen_links.when("routes['resource'] is not None", prio=10 )
    def _gen_links_res(self, routes):
        return [self.index_link_tuple()]   
        
    @gen_links.when("routes['resource'] is not None and routes.get('id',None) is not None", prio=10)
    def _gen_links_obj(self, routes):
        return [  self.show_link_tuple(),
                  self.index_link_tuple(),
                ]