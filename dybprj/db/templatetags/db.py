
from django import template

register = template.Library()

from time import mktime
epoch = lambda dt:mktime(dt.timetuple())

def svgline( vld ):
    return """
       <g> 
         <line x1="%(x1)s" y1="%(y1)s" x2="%(x2)s" y2="%(y2)s" style="stroke:rgb(99,99,99);stroke-width:2"/>
       </g>
   """ % dict(x1=epoch(vld.TIMESTART),y1=vld.SEQNO,x2=epoch(vld.TIMEEND),y2=vld.SEQNO )


register.filter('svgline', svgline )


