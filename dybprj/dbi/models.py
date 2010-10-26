from django.db import models


database_admin_display = ('name','created',)
database_admin_filter = ('name','created',)

class Database(models.Model):
    name = models.CharField(max_length=40)
    created = models.DateTimeField(auto_now_add=True)

    def __unicode__(self):
        return ",".join( ["%s=%s" % ( k, getattr(self,k)) for k in database_admin_display ] )



table_admin_display = ('name','created','db',)
table_admin_filter = ('name','created','db',)

class Table(models.Model):
    name = models.CharField(max_length=40)
    created = models.DateTimeField(auto_now_add=True)
    db = models.ForeignKey(Database)

    class Meta:
        permissions = (
            ('edit_table', 'Can edit table'),
        )
    def __unicode__(self):
        return ",".join( ["%s=%s" % ( k, getattr(self,k)) for k in table_admin_display ] )


