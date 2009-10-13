
from tw.api import Widget, lazystring as _

class Summary(Widget):
    template = "genshi:vdbi.tw.rum.templates.summary"
    params = []
    css_class = "rum-paginator"
    page_size = None
    radius = 5

    def update_params(self, d):
        super(Summary, self).update_params(d)
        query = d.value
        offset = query.offset or 0
        d.count = query.count
 
    def display(self, value, **kw):
        if value is None:
            return ''
        return super(Summary, self).display(value, **kw)

    def render(self, value, **kw):
        if value is None:
            return ''
        return super(Summary, self).render(value, **kw)

