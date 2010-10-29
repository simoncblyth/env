from django.db import models
from django.conf import settings
from django.template.loader import render_to_string
from olive import publish_to_q

def olive_address( obj ):
    """ Provides the OLIVE address string for an object, of form "exchange:key"
          eg "olive:olive.db.table.2.string" 
       This acts as an absolute specification of an instance within a django project 
    """
    key = ".".join([obj._meta.app_label, obj._meta.object_name.lower(), str(obj.pk)])
    return ":".join( [ settings.OLIVE_AMQP_EXCHANGE , settings.OLIVE_KEY_FUNC( key ) ])


from django.contrib.comments.signals import comment_was_posted
def olive_callback(sender, comment, request , **kwa ):
    """ Invoked after POST-ed comments are saved to DB """ 
    addr  = olive_address( comment.content_object )
    publish_to_q( addr , render_to_string( "comments/item.html" , dict( comment=comment ) ))
    pass
comment_was_posted.connect( olive_callback )


