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

from trac.wiki import wiki_to_html, wiki_to_oneliner

STYLE  = 'margin-top: 2px; color:black; background-color:%s; '\
         'border: solid black 1px'
COLOR  = 'white'
LEGEND = 'Items'

def execute(hdf, txt, env):
    lines  = txt.split('\n')
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
    
    return _build_field_set(hdf, '\n'.join(lines[offset:]), env, legend, color)

def _build_field_set(hdf, txt, env, legend, color):
    style = STYLE % color
    html = '<fieldset style="%s"><legend style="%s">' \
           '%s</legend>%s</fieldset>' % \
	  (style,style,wiki_to_oneliner(legend,env),wiki_to_html(txt,env,None))
    return html
