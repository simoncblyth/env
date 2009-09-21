"""Pylons environment configuration"""
import os

from genshi.template import TemplateLoader
from pylons.configuration import PylonsConfig
from sqlalchemy import engine_from_config

import plvdbi.lib.app_globals as app_globals
import plvdbi.lib.helpers
from plvdbi.config.routing import make_map
from plvdbi.model import init_model

def load_environment(global_conf, app_conf):
    """Configure the Pylons environment via the ``pylons.config``
    object
    """
    config = PylonsConfig()
    
    #from pkg_resources import resource_filename
    #vdbi_templates = os.path.abspath(resource_filename('vdbi.rum','templates'))
    
    # Pylons paths
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = dict(root=root,
                 controllers=os.path.join(root, 'controllers'),
                 static_files=os.path.join(root, 'public'),
                 templates=[os.path.join(root, 'templates')])

    # Initialize config with the basic options
    config.init_app(global_conf, app_conf, package='plvdbi', paths=paths)

    config['routes.map'] = make_map(config)
    config['pylons.app_globals'] = app_globals.Globals(config)
    config['pylons.h'] = plvdbi.lib.helpers

    # add the template path from the vdbi app to pylons search list ... for the login/logout functionality 
    vdbi_app = config['pylons.app_globals'].vdbi_app
    tmplpath = paths['templates']
    for p in vdbi_app.config['templating']['search_path']:
        tmplpath.append(p)

    # Create the Genshi TemplateLoader
    config['pylons.app_globals'].genshi_loader = TemplateLoader(
        tmplpath , auto_reload=True)


    # Setup the SQLAlchemy database engine
    engine = engine_from_config(config, 'sqlalchemy.')
    init_model(engine)

    # CONFIGURATION OPTIONS HERE (note: all config options will override
    # any Pylons config options)
    
    return config
