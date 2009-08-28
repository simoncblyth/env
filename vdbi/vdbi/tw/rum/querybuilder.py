from tw.api import lazystring as _
from tw import forms
from tw.forms.validators import UnicodeString

from rum import app, fields
from rum.query import Query
from tw.rum.repeater import JSRepeater

def get_fields():
    try:
        return [(f.name, f.label)
                for f in app.fields_for_resource(app.request.routes['resource'])
                if getattr(f, 'searchable', False)]
    except:
        return []

operators = [
    ("eq", _("is")),
    ("neq", _("is not")),
    ("contains", _("contains")),
    ("startswith", _("begins with")),
    ("endswith", _("ends with")),
    ("lt", _("<")),
    ("gt", _(">")),
    ("lte", _("<=")),
    ("gte", _(">=")),
    ("null", _("is empty")),
    ("notnull", _("is not empty")),
    ]

class ExpressionWidget(forms.FieldSet):
    css_class = "rum-querybuilder-expression"
    template = "genshi:tw.rum.templates.expression"
    fields = [
        forms.SingleSelectField("c", options=get_fields),
        forms.SingleSelectField("o", options=operators),
        forms.TextField("a", validator=UnicodeString),
        ]




class QueryWidget(forms.FieldSet):
    template = "genshi:tw.rum.templates.querybuilder"
    css_class = "rum-query-widget"
    fields = [
        forms.SingleSelectField("o",
            options=[("and", _("AND")), ("or", _("OR"))]
            ),
        JSRepeater("c", widget=ExpressionWidget(), extra=0,
                   add_text=_("Add criteria"), remove_text=_("Remove"))
        ]


def dbi_fields():
    from vdbi.dyb import Enum
    from vdbi import CTX_COLUMNS, CTX_KEYS
    e = Enum()
    ssf = lambda n,k:forms.SingleSelectField(n, options=e.options(k)) 
    return map(ssf, CTX_COLUMNS, CTX_KEYS )

class DbiQueryWidget(forms.FieldSet):
    template = "genshi:vdbi.tw.rum.templates.querybuilder"
    css_class = "rum-query-widget"
    fields = dbi_fields() + [
        forms.SingleSelectField("o", options=[("and", _("AND")), ("or", _("OR"))]),
        JSRepeater("c", widget=ExpressionWidget(), extra=0, add_text=_("Add criteria"), remove_text=_("Remove"))
        ]


class DbiQueryBuilder(forms.TableForm):
    method = "get"
    css_class = "rum-query-builder"
    submit_text = _("Filter DBI records")
    fields = [
        DbiQueryWidget("q", label_text=''),
        ]

    def adapt_value(self, value):
        if isinstance(value, Query):
            value = value.as_dict()
        return value



if __name__=='__main__':
    print DbiQueryWidget.fields
