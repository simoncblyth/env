
from trac.wiki.macros import WikiMacroBase
from trac.wiki.formatter import format_to_html
from genshi.builder import tag 


class FieldSetMacro(WikiMacroBase):
    """FieldSet macro.

       Simply places the macro content into an html fieldset..
       which can avoid the overwide diff problem.
       
       This is a 0.11 only macro 
           
    """

    revision = "$Rev$"
    url = "$URL$"
  
    def expand_macro(self, formatter, name, args):
        """
        """
        fs = tag.fieldset(style="border:none")        
        return fs( format_to_html( self.env, formatter.context, args ) )
    





