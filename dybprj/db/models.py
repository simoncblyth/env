"""
   A model of the model ...   
      * bootstraped at syncdb time 

    django ORM cannot map DBI tables directly
       ... due to DBI usage of composite primary keys in pay tables
       ... it can be made to do so with kludges, but not a valid mapping ... only works in special cases 

   Detailed row level access for plots etc.. will need to use ...
      * SQLAlchemy
      * MySQLdb 

   BUT is useful to have the skeleton of the domain in django ORM as backbone 


"""

from django.db import models
from django.core.urlresolvers import reverse

database_admin_display = ('name','created',)
database_admin_filter = ('name','created',)

class Database(models.Model):
    name = models.CharField(max_length=40)
    created = models.DateTimeField(auto_now_add=True)

    def get_absolute_url(self):
        return reverse( 'db-database' , kwargs=dict(dbname=self.name) ) 

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



column_admin_display = ('name','created','table',)
column_admin_filter = ('name','created','table',)

class Column(models.Model):
    name = models.CharField(max_length=40)
    created = models.DateTimeField(auto_now_add=True)
    table = models.ForeignKey(Table)

    def get_absolute_url(self):
        return reverse( 'db-column' , kwargs=dict(dbname=self.table.db.name , tabname=self.table.name, colname=self.name  ) ) 

    def __unicode__(self):
        return ",".join( ["%s=%s" % ( k, getattr(self,k)) for k in column_admin_display ] )







