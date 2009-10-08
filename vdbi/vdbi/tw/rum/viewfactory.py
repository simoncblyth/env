
#import IPython ; debug_here = IPython.Debugger.Tracer()

from rum import app, fields

from tw.rum import WidgetFactory
from tw.rum.viewfactory import get, inline_actions
from tw.rum import widgets

from vdbi.tw.rum.querybuilder import DbiQueryBuilder, DbiLogin
from vdbi.tw.rum.contextlinks import DbiContextLinks
from vdbi.tw.rum.plotbuilder import  DbiPlotView, JSONLink
from vdbi.tw.rum.grid import dummy

from vdbi.tw.rum.widgets import DbiCollectionLinks


import logging
log = logging.getLogger(__name__)


class DbiWidgetFactory(WidgetFactory):
    def __init__(self, *args, **kwargs ):
        """
            the overrides have to come first, as the first setdefault wins
        """
        rum_widgets = app.config['widgets']
        rum_widgets.setdefault('querybuilder', DbiQueryBuilder())   
        rum_widgets.setdefault('context_links', DbiContextLinks())
        rum_widgets.setdefault('jsonlink', JSONLink())
        rum_widgets.setdefault('plotview', DbiPlotView())
        rum_widgets.setdefault('dbilogin', DbiLogin())
        #print "customized %s " % (repr(self))
        super(DbiWidgetFactory, self).__init__(*args, **kwargs)
        #for k,v in app.config['widgets'].items():print k,v.__class__
    @get.when(
"isinstance(attr, fields.Collection) and action in inline_actions",
            prio=-1
            )
    def _widget_for_collection_in_grid_col(
            self, resource, parent, remote_name, attr, action, args
            ):
        args['field'] = attr
        #debug_here()
        log.debug("collection links rule applied for action %s, resource %r, field %r", action, resource, attr)
        return DbiCollectionLinks



