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
           forms.SingleSelectField('SimFlag', options=ctx.options('SimFlag')),   
           forms.SingleSelectField('Site', options=ctx.options('Site')),
           forms.SingleSelectField('DetectorId' , options=ctx.options('DetectorId')),
           DbiCalendarDateTimePicker('Timestamp'),
        ]

## default=ctx['SimFlag.default']
## default=ctx['Site.default'])
## default=ctx['DetectorId.default']


class DbiContextWidget(forms.FieldSet):
    template = "genshi:vdbi.tw.rum.templates.querybuilder"
    css_class = "rum-query-widget"
    fields =  [
       forms.SingleSelectField("o", options=[("and", _("AND")), ("or", _("OR"))] ),
       JSRepeater("c", widget=DbiExpressionWidget(), extra=0, add_text=_("Add context"), remove_text=_("Remove"))
        ]

class DbiQueryWidget(forms.FieldSet):
    template = "genshi:vdbi.tw.rum.templates.querywidget"
    css_class = "rum-query-widget"
    fields = [
          DbiContextWidget("ctx", label_text=''),
          QueryWidget("xtr", label_text=''),         
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
        #debug_here()
        value = ReContext(value)()
        return value



def xml_parse( txt ):
    from StringIO import StringIO
    demo = StringIO( str(txt) )
    from xml.etree import ElementTree as ET
    t = ET.parse( demo )
    r = t.getroot()
    return r



class WidgetTest(dict):
    xhtml = "http://www.w3.org/1999/xhtml"
    def __init__(self,widget, vls):
        self.widget = widget
        self.vls = vls
        self.root = xml_parse( widget(vls) )   
    
    def __call__(self):
        self.selects_()
        self.inputs_()
        if self.vls != dict(self):
            print "WidgetTest mismatch vls:%s chk:%s " % ( repr(self.vls) , repr(dict(self)) )
        return self
        
    def selects_(self):
        for select in self.root.findall(".//{%s}select" % self.xhtml ):
            id = select.attrib['id']
            opts = select.findall(".//{%s}option" % self.xhtml )
            sopt = [o for o in opts if o.attrib.get('selected',False) ]
            if len(sopt) == 1:
                if id in self.vls:
                    v = sopt[0].attrib['value']
                    try:
                        self[id] = int(v) 
                    except ValueError:
                        self[id] = v
    
    def inputs_(self):
        for input in self.root.findall(".//{%s}input" % self.xhtml ):
            id = input.attrib['id']
            if id in self.vls:
                self[id] = input.attrib['value']
 
     
    


if __name__=='__main__':
    #print DbiQueryWidget.fields
    #dqb = DbiQueryBuilder()
    #print dqb

    #from rum.query import *   
    #q = Query(and_([eq('SITE',1)]))
    #kw = {}
    #d = dqb.prepare_dict(q , kw )   ## this is what gets fed to the template     
    #print dqb(q)   ## see the html
    
    
    
 
    dew = DbiExpressionWidget()
    dew_vls = { 'SimFlag':2 , 'Site':32 , 'DetectorId':7 , 'Timestamp':"2009/09/01 18:39" } 
    dew_test = WidgetTest( dew , dew_vls )()


    dcw = DbiContextWidget()
    dcw_vls = { 'c':[dew_vls] , 'o':"and"  } 
    dcw_test = WidgetTest( dcw , dcw_vls )()
    
    
    
    
    
    

