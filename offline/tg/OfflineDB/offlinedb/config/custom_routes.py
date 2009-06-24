

def custom_setup_routes(self):
    """Setup the default TG2 routes

       Overide this and setup your own routes maps if you want to use
       custom routes.
    """
    from tg import config
    from routes import Mapper
    map = Mapper(directory=config['pylons.paths']['controllers'],
                        always_scan=config['debug'])


    # diverge from the standard object dispatch 
    from offlinedb.controllers.dbi import add_routes
    add_routes(map)

    # Setup a default route for the root of object dispatch
    map.connect('*url', controller='root', action='routes_placeholder')
    config['routes.map'] = map



