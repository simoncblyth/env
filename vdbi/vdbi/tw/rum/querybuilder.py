from tw.api import lazystring as _
from tw import forms
from tw.forms.validators import UnicodeString
from tw.forms.validators import Int


from rum import app, fields
from rum.query import Query
from tw.rum.repeater import JSRepeater

from vdbi import debug_here
#from vdbi.rum.query import ReContext



def get_fields():
    try:
        return [(f.name, f.label)
                for f in app.fields_for_resource(app.request.routes['resource'])
                if getattr(f, 'searchable', False)]
    except:
        return []

## fake for testing
#def get_fields():return [(f,f) for f in ("SITE","WIBBLE","WOBBLE")]
   

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
           forms.SingleSelectField('SimFlag', options=ctx.options('SimFlag'), default=ctx['_default']['SimFlag'], validator=Int() ),   
           forms.SingleSelectField('Site', options=ctx.options('Site') , default=ctx['_default']['Site']   , validator=Int() ),
           forms.SingleSelectField('DetectorId' , options=ctx.options('DetectorId') , default=ctx['_default']['DetectorId'] , validator=Int() ),
           DbiCalendarDateTimePicker('Timestamp'),
        ]



class DbiContextWidget(forms.FieldSet):
    template = "genshi:vdbi.tw.rum.templates.querybuilder"
    css_class = "rum-query-widget"
    fields =  [
       forms.HiddenField("o", default="ctx_" ),    
       forms.SingleSelectField("a", options=[("and", _("AND")), ("or", _("OR"))] ),
       JSRepeater("c", widget=DbiExpressionWidget(), extra=0, add_text=_("Add context"), remove_text=_("Remove"))
        ]



## if could make an anonymous widget ... would be able to do widget algebra ?
class DbiQueryWidget(forms.FieldSet):
    template = "genshi:vdbi.tw.rum.templates.querywidget"
    css_class = "rum-query-widget"
    fields = [
                 DbiContextWidget( "ctx", label_text=''),
                 QueryWidget( "xtr", label_text=''), 
             ]
  
  
  
from vdbi.rum.query import _vdbi_uncast       
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
            value = _vdbi_uncast(value)
        return value



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
    
    ## nothing was getting thru to values ... 
    ## due to the ReContext always being done ... changed to only being done for Query values 
    
    #from rum.query import *   
    #q = Query(and_([eq('VSITE', u'1'), eq('VSIM', u'1'), eq('VSUB', u'0'), lt('VSTART', u'2009/09/01 19:12'), gt('VEND', u'2009/09/01 19:12')]))
      
    #dqq =  DbiQueryBuilder("dqq")
    #dqq_v = q
    #dqq_t = WidgetTest( dqq , dqq_v  )()  
    
    #kw = {}
    #d = dqb.prepare_dict(q , kw )   ## this is what gets fed to the template
    
    ## pull out what the adapt_value is doing ... converting the query into a dict 
    ## the dict created assumes the original Rum widget layout ... so use ReContext 
    ## to rejig the dict to fit into the new widget layout
    
    #qd = q.as_dict()
    #qr = ReContext(qd)
    #qv = qr()
    #qt = WidgetTest( dqq , qd  )()    ## only timestamp getting thru to context, fixed by changing from string to int ... now all getting thru 
    
    
    #qxt = and_( [  eq(u'RING', u'2') , q.expr ])
    #qq = q.clone( expr=qxt )
    #qqd = qq.as_dict()
    #qqr = ReContext(qqd)
    #qqv = qqr()
    #qqt = WidgetTest( dqq , qqv )()
    
    

    
    
    

