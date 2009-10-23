from tw.api import lazystring as _
from tw import forms
from tw.forms.validators import UnicodeString, Int, NotEmpty
import tw.dynforms as twd

from rum import app, fields
from rum.query import Query

from vdbi.tw.rum.repeater import DbiJSRepeater
from tw.rum import widgets
from vdbi.dbg import debug_here
from vdbi import get_default_x, get_default_y
from vdbi.rum.param import offset, limit, width, height, present


def get_fields():
    try:
        return [(f.name, f.label)
                for f in app.fields_for_resource(app.request.routes['resource'])
                if getattr(f, 'searchable', False)]
    except:
        return []


def get_plotable_fields():
    try:
        return [(f.name, f.label)
                for f in app.fields_for_resource(app.request.routes['resource'])
                if getattr(f, 'plotable', False)]
    except:
        return []
 


def get_default_x_():
    routes=app.request.routes
    return get_default_x(routes['resource'])

def get_default_y_():
    routes=app.request.routes
    return get_default_y(routes['resource'])




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
        forms.SingleSelectField("c", options=get_fields, default=get_default_y_ ),
        forms.SingleSelectField("o", options=operators, default=operators[0][0] ),
        forms.TextField("a", validator=UnicodeString),
        ]




class PlotSeriesWidget(forms.FieldSet):
    css_class = "rum-querybuilder-expression"
    template = "genshi:vdbi.tw.rum.templates.plotexpression"
    fields = [
        forms.SingleSelectField("x", options=get_plotable_fields, default=get_default_x_ ),
        forms.SingleSelectField("y", options=get_plotable_fields, default=get_default_y_ ),
        ]



##  forms.SingleSelectField("a",options=[("table", _("Table")), ("plot", _("Plot")), ("both", _("Plot+Table"))]),





 


class PlotParametersWidget(forms.TableFieldSet):
    css_class = "rum-query-widget"
    fields = [
         forms.TextField("limit", **limit ),
         forms.TextField("offset", **offset ),
         forms.TextField("width",  **width ),
         forms.TextField("height", **height ),
             ]


class PlotWidget(forms.FieldSet):
    template = "genshi:vdbi.tw.rum.templates.plotwidget"
    css_class = "rum-query-widget"
    fields = [
        PlotParametersWidget("param", legend="Plot Parameters"),
        DbiJSRepeater("c", widget=PlotSeriesWidget(), extra=0,add_text=_("Add plot series"), remove_text=_("Remove")),
        forms.HiddenField("o", default="plt_" ), 
        ]


##class DbiCalendarDateTimePicker(forms.CalendarDateTimePicker):


from tw.api import JSLink
calendar_js = JSLink( modname='tw.forms', filename='static/calendar/calendar.js')
calendar_setup = JSLink( modname='vdbi.tw.rum', filename='static/calendar/calendar_setup.js')

class DbiCalendarDateTimePicker(widgets.RumCalendarDateTimePicker):
    css_class = "rum-querybuilder-expression"
    javascript = [calendar_js, calendar_setup]
    
    def __init__(self, *args, **kw):
        #print "DbiCalendarDateTimePicker.__init__ %s %s " % ( repr(args), repr(kw))
        super(DbiCalendarDateTimePicker, self).__init__(*args, **kw)
    
    def update_params(self, d):
        super(DbiCalendarDateTimePicker, self).update_params(d)
        #print "DbiCalendarDateTimePicker.update_params %s  " % (  repr(d))

    

class QueryWidget(forms.FieldSet):
    template = "genshi:vdbi.tw.rum.templates.querybuilder"
    css_class = "rum-query-widget"
    fields = [
        forms.SingleSelectField("o",options=[("and", _("AND")), ("or", _("OR"))]),
        DbiJSRepeater("c", widget=ExpressionWidget(), extra=0,add_text=_("Add criteria"), remove_text=_("Remove")),
        forms.HiddenField("a", default="xtr_" ), 
        ]


from vdbi.dyb import ctx
class DbiExpressionWidget(forms.FieldSet):
    template = "genshi:vdbi.tw.rum.templates.expression"
    css_class = "rum-querybuilder-expression"
    fields =  [
           forms.SingleSelectField('SimFlag', options=ctx.options('SimFlag'), default=ctx['_default']['SimFlag'], validator=Int(min=0) ),   
           forms.SingleSelectField('Site', options=ctx.options('Site') , default=ctx['_default']['Site']   , validator=Int(min=0) ),
           forms.SingleSelectField('DetectorId' , options=ctx.options('DetectorId') , default=ctx['_default']['DetectorId'] , validator=Int(min=0) ),
           DbiCalendarDateTimePicker('Timestamp', validator=NotEmpty),
        ]


## need the DbiJSRepeater in order to do the calendar hookup in repetitions
class DbiContextWidget(forms.FieldSet):
    template = "genshi:vdbi.tw.rum.templates.querybuilder"
    css_class = "rum-query-widget"
    fields =  [
       forms.HiddenField("o", default="ctx_" ),    
       forms.SingleSelectField("a", options=[("and", _("AND")), ("or", _("OR"))] ),
       DbiJSRepeater("c", widget=DbiExpressionWidget(), extra=0, add_text=_("Add context"), remove_text=_("Remove"))
        ]





class DbiQueryWidget(twd.HidingTableFieldSet):
    template = "genshi:vdbi.tw.rum.templates.querywidget"
    css_class = "rum-query-widget"
    fields = [
                 DbiContextWidget( "ctx", label_text=''),
                 QueryWidget( "xtr", label_text=''), 
                 twd.HidingCheckBoxList('present', **present ),
                 PlotWidget("plt", label_text=''),
             ]

    
from vdbi.rum.query import _vdbi_widget       
class DbiQueryBuilder(forms.TableForm):
    method = "get"
    css_class = "rum-query-builder"
    submit_text = _("UPDATE")
    fields = [
          DbiQueryWidget("q", label_text=''), 
        ]

    def adapt_value(self, value):
        if isinstance(value, Query):
            print "DbiQueryBuilder.adapt_value query : %s " % `value`
            value = _vdbi_widget(value.as_dict())
            print "DbiQueryBuilder.adapt_value 4widget:  %s " % `value`
        return value




## used for scraping ... for static from outside app
class DbiLogin(forms.TableForm):
    method = "post"
    submit_text = _("Login")
    action = "%s"
    fields = [
        forms.TextField("username"),
        forms.PasswordField("password"),
    ]







def xml_parse( txt ):
    from StringIO import StringIO
    demo = StringIO( str(txt) )
    from xml.etree import ElementTree as ET
    t = ET.parse( demo )
    r = t.getroot()
    return r

def xhtml_find( a, elemname ):
    xhtml = "http://www.w3.org/1999/xhtml"
    return a.findall(".//{%s}%s" % (xhtml, elemname) )

class WidgetTest(list):
    
    def __init__(self, widget, value, **kw ):
        self.widget = widget
        self.value = value
        self.kw = kw
        self.root = xml_parse( str(self) )   
    
    def __str__(self):
        return self.widget( self.value , **self.kw )
    
    def __call__(self):
        self.selects_()
        self.inputs_()
        self.check_()
        return self
    
    def find_(self, elemname ):
        return xhtml_find(self.root, elemname )
        
    def selects_(self):
        for select in self.find_('select'):
            id = select.attrib['id']
            opts = xhtml_find( select , "option")
            sopt = [o for o in opts if o.attrib.get('selected',False) ]
            if len(sopt) == 1:
                v = sopt[0].attrib['value']
                try:
                    iv = int(v)
                except ValueError:
                    iv = v
                self.append( { id:iv } ) 
    
    def inputs_(self):
        for input in self.find_('input'):
            id = input.attrib['id']
            self.append( { id:input.attrib['value'] })
  
    def check_(self):      
        print "WidgetTest check value:%s xmlchk:%s " % ( repr(self.value) , repr(list(self)) )  
    


if __name__=='__main__':
    #print DbiQueryWidget.fields


    from vdbi.app import setup_logging
    setup_logging()
    
    
    from xml.etree import ElementTree as ET
 
    dew = DbiExpressionWidget('c')
    dew_v = { 'SimFlag':2 , 'Site':32 , 'DetectorId':7 , 'Timestamp':"2009/09/01 18:39" } 
    dew_t = WidgetTest( dew , dew_v )()

    dcw = DbiContextWidget("ctx")
    dcw_v = { 'c':[dew_v] , 'o':"and"  } 
    dcw_t = WidgetTest( dcw , dcw_v )()
     
    ew = ExpressionWidget("c")
    ew_v = {'c':"SITE" , 'o':"eq" , 'a':1 }
    ew_t = WidgetTest( ew , ew_v )()
       
    qw = QueryWidget('xtr')
    qw_v = {  'o':"and" , 'c':[ew_v] }
    qw_t = WidgetTest( qw , qw_v )()
    
    dqw = DbiQueryWidget("q")
    dqw_v = { 'ctx':dcw_v , 'xtr':qw_v }
    dqw_t = WidgetTest( dqw , dqw_v )()    ## working 
    
    dqb = DbiQueryBuilder("dqb")
    dqb_v = { 'q':dqw_v }
    dqb_t = WidgetTest( dqb , dqb_v , adapt=False )()       ## this adapt seems not to be honoured
    
 
    

    
    
    

