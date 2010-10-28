from django.db import models
from django.conf import settings
from django.template.loader import render_to_string

from olive import publish_to_q

def olive_key( instance ):
    return  settings.OLIVE_KEY_FUNC( ".".join([instance.content_type.app_label, instance.content_type.model, str(instance.content_object.pk)]))

from django.contrib.comments.signals import comment_was_posted
def olive_callback(sender, comment, request , **kwa ):
    """
        This is is invoked after POST-ed comments are saved to DB 

        With 
           * pre-existing exchange "abt" 
           * keys of form olive.#.string
        succeeds to get messages to  pika-consume        

    """ 
    key  = olive_key( comment )
    body = render_to_string( "comments/item.html" , dict( comment=comment ) )
    publish_to_q( settings.OLIVE_EXCHANGE, key , body ) 
    pass
comment_was_posted.connect( olive_callback )


