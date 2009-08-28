
from rum import app
from tw.rum import WidgetFactory
from vdbi.tw.rum.querybuilder import DbiQueryBuilder

class DbiWidgetFactory(WidgetFactory):
    def __init__(self, *args, **kwargs ):
        print "customized %s " % (repr(self))
        rum_widgets = app.config['widgets']
        rum_widgets.setdefault('querybuilder', DbiQueryBuilder())   ## this has to come first, as the first setdefault wins
        super(DbiWidgetFactory, self).__init__(*args, **kwargs)



