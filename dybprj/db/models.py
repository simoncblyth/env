from django.db import models
from django.core.urlresolvers import reverse

database_admin_display = ('name','created',)
database_admin_filter = ('name','created',)


class Database(models.Model):
    name = models.CharField(max_length=40)
    created = models.DateTimeField(auto_now_add=True)

    def get_absolute_url(self):
        return reverse( 'db-detail' , kwargs=dict(dbname=self.name) ) 

    def __unicode__(self):
        return ",".join( ["%s=%s" % ( k, getattr(self,k)) for k in database_admin_display ] )



table_admin_display = ('name','created','db',)
table_admin_filter = ('name','created','db',)

class Table(models.Model):
    name = models.CharField(max_length=40)
    created = models.DateTimeField(auto_now_add=True)
    db = models.ForeignKey(Database)

    def get_absolute_url(self):
        return reverse( 'db-table' , kwargs=dict(dbname=self.db.name , tabname=self.name ) ) 

    class Meta:
        permissions = (
            ('edit_table', 'Can edit table'),
        )
    def __unicode__(self):
        return ",".join( ["%s=%s" % ( k, getattr(self,k)) for k in table_admin_display ] )








