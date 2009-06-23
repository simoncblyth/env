
from tg.controllers import DecoratedController
from tg.exceptions import HTTPException

class RoutingController(DecoratedController):
    """
         http://simplestation.com/locomotion/routes-in-turbogears2/

    """
    def _perform_call(self, func, args):
        if not args:
            args = {}
 
        try:
            aname = str(args.get('action', 'lookup'))
            controller = getattr(self, aname)
 
            # If these are the __before__ or __after__ methods, they will have no decoration property
            # This will make the default DecoratedController._perform_call() method choke
            # We'll handle them just like TGController handles them.
            func_name = func.__name__
            if func_name == '__before__' or func_name == '__after__':
                if func_name == '__before__' and hasattr(controller.im_class, '__before__'):
                    return controller.im_self.__before__(*args)
                if func_name == '__after__' and hasattr(controller.im_class, '__after__'):
                    return controller.im_self.__after__(*args)
                return
            else:
                controller = func
                params = args
                remainder = ''
 
                result = DecoratedController._perform_call(
                    self, controller, params, remainder=remainder)
 
        except HTTPException, httpe:
            result = httpe
            # 304 Not Modified's shouldn't have a content-type set
            if result.status_int == 304:
                result.headers.pop('Content-Type', None)
            result._exception = True
        return result





def custom_setup_routes(self):
    """Setup the default TG2 routes

       Overide this and setup your own routes maps if you want to use
       custom routes.
    """
    from tg import config
    from routes import Mapper
    map = Mapper(directory=config['pylons.paths']['controllers'],
                        always_scan=config['debug'])

    map.connect( 'dbi', '/dbi/{table}', controller='dbi', action='noodles' ) 

    # Setup a default route for the root of object dispatch
    map.connect('*url', controller='root', action='routes_placeholder')
    config['routes.map'] = map





