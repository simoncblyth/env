import logging

from pylons import request, response, session, tmpl_context as c
from pylons.controllers.util import abort, redirect_to

from plvdbi.lib.base import BaseController, render

log = logging.getLogger(__name__)

class AuthController(BaseController):

    def index(self):
        # Return a rendered template
        #return render('/auth.mako')
        # or, return a response
        return 'Hello World'
        
    def logout(self):
        return 'You Are Now Logged Out '
