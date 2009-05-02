'''
A wiki-processor for encapsulating wiki text inside a box.
The box will have a legend and a modifiable color.

Example:

{{{
#!LegendBox
#!color: blue
#!legend: My Title
Here comes the actual text that comes inside the box.
You can even use wiki-formatting in here.
}}}

You can modify the default COLOR, LEGEND and the STYLE in this plugin file.

'''

from trac.wiki.macros import WikiMacroBase
from trac.wiki.formatter import format_to_html
from genshi.builder import tag 

STYLE  = 'margin-top: 2px; color:black; background-color:%s; '\
         'border: solid black 1px'
COLOR  = 'white'
LEGEND = 'Items'

class LegendBoxMacro(WikiMacroBase):
    def expand_macro(self, formatter, name, args):
        """Return some output that will be displayed in the Wiki content.

        `name` is the actual name of the macro 
        `args` is the text enclosed in parenthesis at the call of the macro.
          Note that if there are ''no'' parenthesis (like in, e.g.
          [[HelloWorld]]), then `args` is `None`.
        """

        lines  = args.split('\n')
        color  = COLOR
        legend = LEGEND
        offset = 0
        for l in lines[:2]:
            if l.startswith('#!color'):
                color = l.split(':',1)[-1]
                offset += 1
            elif l.startswith('#!legend'):
                legend = l.split(':',1)[-1]
                offset += 1

        style = STYLE % color
        return tag.fieldset(style=style)( tag.legend(style=style)(legend), format_to_html(self.env, formatter.context, "\n".join(lines[offset:]) ) )        
  


