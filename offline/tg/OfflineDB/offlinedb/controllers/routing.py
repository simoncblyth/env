
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






