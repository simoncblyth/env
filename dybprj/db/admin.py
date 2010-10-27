
from django.contrib import admin
from django.db import connections

from models import *

class TableAdmin(admin.ModelAdmin):
    date_heirarchy = 'created'
    list_display = table_admin_display
    list_filter  = table_admin_filter
admin.site.register( Table, TableAdmin )


def ingest_all_tables(modeladmin, request, queryset):
    """
         TODO : deletion of tables no longer in the db 
    """
    for db in queryset:
        conn = connections[db.name]
        cursor = conn.cursor()
        cursor.execute("SHOW TABLES")
        for row in cursor.fetchall():
            Table.objects.get_or_create(name=row[0], db=db)
ingest_all_tables.short_description = "Ingest/Update table names from selected databases "

## StackedInline for a different template
##class TableInline(admin.TabularInline):
class TableInline(admin.StackedInline):
    model = Table

class DatabaseAdmin(admin.ModelAdmin):
    date_heirarchy = 'created'
    list_display = database_admin_display
    list_filter  = database_admin_filter
    actions = (ingest_all_tables,)
    inlines = (TableInline,)


admin.site.register( Database, DatabaseAdmin )



