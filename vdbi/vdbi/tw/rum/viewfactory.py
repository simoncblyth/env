
from tw.rum import WidgetFactory

class DbiWidgetFactory(WidgetFactory):
    def __init__(self, *args, **kwargs ):
        print "customized %s " % (repr(self))
        super(DbiWidgetFactory, self).__init__(*args, **kwargs)


