import logging

from pylons import request, response, session, tmpl_context as c
from pylons.controllers.util import abort, redirect_to
from pylons import app_globals  

from plvdbi.lib.base import BaseController, render

log = logging.getLogger(__name__)


class AuthController(BaseController):

    def index(self):
        return 'Whaddya do to get here ?'
        
    def logout(self):
        return redirect_to(controller='dbi',path_info='')

        

        
    
