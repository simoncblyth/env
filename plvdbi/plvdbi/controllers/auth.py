import logging

from pylons import request, response, session, tmpl_context as c
from pylons.controllers.util import abort, redirect_to
from pylons import app_globals  

from plvdbi.lib.base import BaseController, render

log = logging.getLogger(__name__)

class FlashDummy:
    def render(self,*args):
        return '<div></div>'



class AuthController(BaseController):

    def index(self):
        # Return a rendered template
        #return render('/auth.mako')
        # or, return a response
        return 'Hello World'
        
    def logout(self):
        return 'You Are Now Logged Out '
      
    def tmpltest(self):
        vapp = app_globals.vdbi_app
        extra_vars = { 
           'widgets':vapp.config['widgets'],
           'master_template':"master.html",
           'resources':[],
           'url_for':vapp.url_for,
           'flash':FlashDummy(),
        } 
        return render("dbilogin.html", extra_vars=extra_vars )
        
    def environ(self):
        result = '<html><body><h1>Environ</h1>'
        for key, value in request.environ.items():
            result += '%s: %r <br />'%(key, value)
        result += '</body></html>'
        return result
        
    def exception(self):
        raise Exception('Just testing the interactive debugger!')
        
    
