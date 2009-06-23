
from tg import expose
from offlinedb.controllers.routing import RoutingController

class DbiController(RoutingController):
    """
    """
    #@expose('offlinedb.templates.dbi.index')
    @expose()
    def index(self):
        """Handle the list of dbi tables."""
        #return dict(page='index')
        return 'dbi controller table list'


