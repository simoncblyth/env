
from rum import app
from tw.rum import WidgetFactory

from vdbi.tw.rum.querybuilder import DbiQueryBuilder
from vdbi.tw.rum.contextlinks import DbiContextLinks
from vdbi.tw.rum.plotbuilder import  DbiPlotView
from vdbi.tw.rum.grid import dummy


class DbiWidgetFactory(WidgetFactory):
    def __init__(self, *args, **kwargs ):
        """
            the overrides have to come first, as the first setdefault wins
        """
        rum_widgets = app.config['widgets']
        rum_widgets.setdefault('querybuilder', DbiQueryBuilder())   
        rum_widgets.setdefault('context_links', DbiContextLinks())
        rum_widgets.setdefault('plotview', DbiPlotView())
        print "customized %s " % (repr(self))
        super(DbiWidgetFactory, self).__init__(*args, **kwargs)

        for k,v in app.config['widgets'].items():
            print k,v.__class__

