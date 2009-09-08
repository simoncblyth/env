from rum import app
from tw.rum import WidgetFactory

from rumtute.tw.rum.querybuilder import TuteQueryBuilder


class TuteWidgetFactory(WidgetFactory):
    def __init__(self, *args, **kwargs ):
        """
            the overrides have to come first, as the first setdefault wins
        """
        rum_widgets = app.config['widgets']
        rum_widgets.setdefault('querybuilder', TuteQueryBuilder())   
        
        print "customized %s " % (repr(self))
        super(TuteWidgetFactory, self).__init__(*args, **kwargs)

        for k,v in app.config['widgets'].items():
            print k,v.__class__






