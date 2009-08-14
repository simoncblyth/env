from django.db import models

class Engine(models.Model):
    """
        http://www.sqlalchemy.org/docs/05/dbengine.html
    """
    name  = models.CharField(max_length=50)
    dburl = models.CharField(max_length=200)
    last  = models.DateTimeField('last look', blank=True, null=True , editable=False )

    def __unicode__(self):
        return "<Engine %s %s >" % ( self.name, self.dburl )

