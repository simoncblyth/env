from tw.api import lazystring as _
from tw import forms
from rum.query import Query
from tw.rum.querybuilder import QueryWidget

class TuteQueryBuilder(forms.TableForm):
    method = "get"
    css_class = "rum-query-builder"
    submit_text = _("Filter Tute records")
    fields = [
          QueryWidget("q", label_text=''), 
        ]

    def adapt_value(self, value):
        if isinstance(value, Query):
            value = value.as_dict()
        return value


