
from rum import exceptions
from tw.forms.validators import UnicodeString, Int, NotEmpty
import logging
log = logging.getLogger(__name__)


class Param(dict):
    def __call__(self, s ):
        try:
            value = self['validator'].to_python(s)
        except exceptions.Invalid:
            log.error("ignoring invalid parameter \"%s\"  \"%s\" " % (s , self['label_text']  ) )
            value = self['default']
        except:
            log.error("ignoring unexpected parameter \"%s\"  \"%s\" " % (s , self['label_text']  ) )
            value = self['default']
        return value
        
    
limit = Param({
    'validator':Int(not_empty=True,min=1,max=1000),
    'default':500 ,
    'size':5,
    'label_text':"Entries Limit [1:1000]",
})
            
offset = Param({
    'validator':Int(not_empty=True,min=0),
    'default':0,
    'size':5,
    'label_text':"Entry Offset [>=0]",
})    
    
width = Param({
     'validator':Int(not_empty=True,min=300,max=1000),
     'default':600,
     'size':5,
     'label_text':"Pixel Width [300:1000]",
})
  
height = Param({
     'validator':Int(not_empty=True,min=300,max=1000),
     'default':400,
     'size':5,
     'label_text':"Pixel Height [300:1000]",
})  

