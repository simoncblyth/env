
from django.contrib import admin
from models import *

class RunAdmin(admin.ModelAdmin):
    date_heirarchy = 'start'
    list_display = run_admin_display
    list_filter  = run_admin_filter
admin.site.register( Run, RunAdmin )



