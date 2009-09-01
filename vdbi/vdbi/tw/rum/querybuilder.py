from tw.api import lazystring as _
from tw import forms
from tw.forms.validators import UnicodeString

from rum import app, fields
from rum.query import Query
from tw.rum.repeater import JSRepeater

from vdbi import debug_here
from vdbi.rum.query import ReContext



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

class DbiCalendarDateTimePicker(forms.CalendarDateTimePicker):
    css_class = "rum-querybuilder-expression"
    

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


from vdbi.dyb import ctx
class DbiExpressionWidget(forms.FieldSet):
    template = "genshi:vdbi.tw.rum.templates.expression"
    css_class = "rum-querybuilder-expression"
    fields =  [
           forms.SingleSelectField('SimFlag', options=ctx.options('SimFlag'), default=ctx['SimFlag.default']),
           forms.SingleSelectField('Site', options=ctx.options('Site') ,  default=ctx['Site.default']),
           forms.SingleSelectField('DetectorId' , options=ctx.options('DetectorId'), default=ctx['DetectorId.default']),
           DbiCalendarDateTimePicker('Timestamp'),
        ]

class DbiContextWidget(forms.FieldSet):
    template = "genshi:vdbi.tw.rum.templates.querybuilder"
    css_class = "rum-query-widget"
    fields =  [
       forms.SingleSelectField("o",
           options=[("and", _("AND")), ("or", _("OR"))]
           ),
       JSRepeater("c", widget=DbiExpressionWidget(), extra=0,
                  add_text=_("Add context"), remove_text=_("Remove"))
        ]
 

        
class DbiQueryBuilder(forms.TableForm):
    method = "get"
    css_class = "rum-query-builder"
    submit_text = _("Filter DBI records")
    fields = [
         DbiContextWidget("ctx", label_text=''),
         QueryWidget("q", label_text=''),
        ]

    def adapt_value(self, value):
        if isinstance(value, Query):
            value = value.as_dict()
        value = ReContext(value)()
        return value



if __name__=='__main__':
    print DbiQueryWidget.fields
