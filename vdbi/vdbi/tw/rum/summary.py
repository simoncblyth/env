
from tw.api import Widget, lazystring as _
from vdbi.rum.urls import json_url, csv_url

class Summary(Widget):
    template = "genshi:vdbi.tw.rum.templates.summary"
    params = ['count','json','csv']
    css_class = "rum-paginator"
    page_size = None
    radius = 5

    def update_params(self, d):
        super(Summary, self).update_params(d)
        query = d.value
        d.count = query.count
        d.json = json_url(d)
        d.csv  = csv_url(d)
        
    def display(self, value, **kw):
        if value is None:
            return ''
        return super(Summary, self).display(value, **kw)

    def render(self, value, **kw):
        if value is None:
            return ''
        return super(Summary, self).render(value, **kw)

