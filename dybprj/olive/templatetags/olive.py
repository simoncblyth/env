from django import template
from django.conf import settings
from dybprj.olive.models import olive_address

register = template.Library()

"""
    inclusion_tag usage ...
        {% olive_script obj %}
    the comment list for the argument object eg "table" 
    is live updated via the olive server
"""
def olive_script( obj ):
    return dict( olive_address=olive_address(obj) , OLIVE_SERVER_HOST=settings.OLIVE_SERVER_HOST, OLIVE_SERVER_PORT=settings.OLIVE_SERVER_PORT )
register.inclusion_tag('olive/olive_script.js')(olive_script)








