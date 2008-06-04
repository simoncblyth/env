
from trac.wiki.macros import WikiMacroBase
from trac.wiki.formatter import format_to_html
from genshi.builder import tag 


class FieldSetMacro(WikiMacroBase):
    """Simple HelloWorld macro.

    Note that the name of the class is meaningful:
     - it must end with "Macro"
     - what comes before "Macro" ends up being the macro name

    The documentation of the class (i.e. what you're reading)
    will become the documentation of the macro, as shown by
    the !MacroList macro (usually used in the WikiMacros page).
    """

    revision = "$Rev$"
    url = "$URL$"
  
    def expand_macro(self, formatter, name, args):
        """Return some output that will be displayed in the Wiki content.

        `name` is the actual name of the macro (no surprise, here it'll be
        `'HelloWorld'`),
        `args` is the text enclosed in parenthesis at the call of the macro.
          Note that if there are ''no'' parenthesis (like in, e.g.
          [[HelloWorld]]), then `args` is `None`.
        """
        fs = tag.fieldset(style="border:none")        
        return fs( format_to_html( self.env, formatter.context, args ) )
    




