# http://docs.djangoproject.com/en/dev/ref/contrib/admin/#ref-contrib-admin

from django.contrib import admin
from dybsite.dbi.models import Engine

class EngineAdmin(admin.ModelAdmin):
    list_display = ( "name" , "dburl" , "last" )

admin.site.register(Engine, EngineAdmin)

