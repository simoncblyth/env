"""Pylons middleware initialization"""
from beaker.middleware import SessionMiddleware
from paste.cascade import Cascade
from paste.registry import RegistryManager
from paste.urlparser import StaticURLParser
from paste.deploy.converters import asbool
from pylons.middleware import ErrorHandler, StatusCodeRedirect
from pylons.wsgiapp import PylonsApp
from routes.middleware import RoutesMiddleware

import authkit.authenticate
import authkit.authorize
from authkit.permissions import ValidAuthKitUser


from plvdbi.config.environment import load_environment



def config_authkit( app_conf ):
    import os
    from pkg_resources import resource_filename
    vdbi_templates = os.path.abspath(resource_filename('vdbi.rum','templates'))    
    from env.base.private import Private
    priv = Private()
    app_conf['authkit.setup.method'] = 'form, cookie'
    app_conf['authkit.form.authenticate.user.type'] = 'authkit.users:UsersFromFile'
    app_conf['authkit.form.authenticate.user.data'] = priv('VDBI_USERS_PATH')
    app_conf['authkit.form.template.file'] = os.path.join( vdbi_templates , 'login.html')
    app_conf['authkit.form.action'] = '/dbi/' 
    app_conf['authkit.cookie.secret'] = priv('VDBI_COOKIE_SECRET')
    app_conf['authkit.cookie.signoutpath'] = '/auth/logout' 
    print "app_conf %s " % repr(app_conf)


def make_app(global_conf, full_stack=True, static_files=True, **app_conf):
    """Create a Pylons WSGI application and return it

    ``global_conf``
        The inherited configuration for this application. Normally from
        the [DEFAULT] section of the Paste ini file.

    ``full_stack``
        Whether this application provides a full WSGI stack (by default,
        meaning it handles its own exceptions and errors). Disable
        full_stack when this application is "managed" by another WSGI
        middleware.

    ``static_files``
        Whether this application serves its own static files; disable
        when another web server is responsible for serving them.

    ``app_conf``
        The application's local configuration. Normally specified in
        the [app:<name>] section of the Paste ini file (where <name>
        defaults to main).

    """
   
    static_files = False 
    config_authkit(app_conf)
   
    # Configure the Pylons environment
    config = load_environment(global_conf, app_conf)

    # The Pylons WSGI app
    app = PylonsApp(config=config)

    # Routing/Session/Cache Middleware
    app = RoutesMiddleware(app, config['routes.map'])
    app = SessionMiddleware(app, config)

    # CUSTOM MIDDLEWARE HERE (filtered by error handling middlewares)

    if asbool(full_stack):
        # Handle Python exceptions
        app = ErrorHandler(app, global_conf, **config['pylons.errorware'])
        
        # Authorization     http://pylonsbook.com/en/1.0/authentication-and-authorization.html
        permission = ValidAuthKitUser()
        app = authkit.authorize.middleware(app, permission)

        # Authentication handling intercepting 401, 403
        app = authkit.authenticate.middleware(app, app_conf)

        # Display error documents for 401, 403, 404 status codes (and
        # 500 when debug is disabled)
        if asbool(config['debug']):
            errcodes = [401, 403, 404]         ## default is (400, 401, 403, 404)
        else:
            errcodes = [401, 403, 404, 500]  

        app = StatusCodeRedirect(app, errcodes)   

        



    # Establish the Registry for this application
    app = RegistryManager(app)

    if asbool(static_files):
        # Serve static files
        static_app = StaticURLParser(config['pylons.paths']['static_files'])
        app = Cascade([static_app, app])
    app.config = config
    return app
