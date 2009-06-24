
class Arg(dict):
    def __init__(self, **kwa ):[setattr(self,k,v) for k,v in kwa.items()]

class Args(list):
    def pattern(self):
        return "/".join(["{%s}" % a.name for a in self]) 
    def requirements(self):
        reqs = {}
        for a in self:
            v = getattr(a,'requirements',None)
            if v:
                 reqs[a.name] = v 
        return reqs
    def names(self):
        return [a.name for a in self]
    def __repr__(self):
        return "<Args %s %s >" % ( self.pattern() , self.requirements() )

def make_args():
    return Args( [ Arg(name='table', requirements=r'\S*'), 
                   Arg(name='year',  requirements=r'\d{2,4}'), 
                   Arg(name='month', requirements=r'\d{1,2}'), 
                   Arg(name='day',   requirements=r'\d{1,2}'),
                 ]  )   
                               

fields = make_args()



from tg import expose
from offlinedb.controllers.routing import RoutingController
from offlinedb.model import DBSession


class DbiController(RoutingController):
    """
    """
    #@expose('offlinedb.templates.dbi.index')
    @expose()
    def index(self):
        """Handle the list of dbi tables."""
        #return dict(page='index')
        return 'dbi controller table list'

    @expose()
    def debug_args(self, *args, **kwa ):
        return "\n".join( ["%s:%s" % (k,v) for k,v in kwa.items() if k in fields.names()] )

    @expose()
    def view_table(self, *args, **kwa ):
        table = kwa.get('table', None)
        from sqlalchemy.exceptions import NoSuchTableError
        if not(table):return "table not specifieed %s " % kwa 
        try:
            from offlinedb.model.dbi import dbi_
            pair = dbi_.pair(table) 
            return "found paired table:%s " % table
        except NoSuchTableError:
            return "no such table:%s " % table



def add_routes(map):
    #map.connect( 'dbi', '/dbi/{table}/{year}/{month}/{day}', controller='dbi', action='view_table' , requirements=dict(year='\d{2,4}', month='\d{1,2}')) 
    map.connect( 'dbi' , '/dbi/%s' % fields.pattern() , controller='dbi' , action='view_table' , requirements=fields.requirements() )


if __name__=='__main__':
    print fields


