
import datetime
from django.db import models

from multidb.util.manager import MultiDBManager 


class Post(models.Model):
    title = models.TextField()
    body = models.TextField()
    date_submitted = models.DateTimeField(default=datetime.datetime.now)
    
    objects = MultiDBManager( "primary" )

class Link(models.Model):
    url = models.URLField()
    description = models.TextField(null=True, blank=True)
    date_submitted = models.DateTimeField(default=datetime.datetime.now)

    objects = MultiDBManager( "secondary" )



