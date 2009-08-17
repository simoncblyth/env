"""
   Attempt to get an out-of-admin changelist to work
   with sqlalchemy underpinnings

"""

## http://blog.stiod.com/2008/12/16/reaproveitando-a-inteface-do-admin-do-django/

from django.shortcuts import render_to_response
from django.template import RequestContext
from django.utils.text import ugettext_lazy as _
from django.contrib.auth.decorators import login_required
from mysearch.profiles.models import Profile
from mysearch.profiles.admin import ProfileAdmin
from django.contrib import admin
from django.contrib.admin.views.main import ChangeList
 
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
 
@login_required
def index(request):
    admin_model = admin.site._registry[Profile]
    admin_model.admin_site.root_path = request.path
    cl = ChangeList(
        request,
        Profile,
        ProfileAdmin.list_display,
        ProfileAdmin.list_display_links,
        ProfileAdmin.list_filter,
        ProfileAdmin.date_hierarchy,
        ProfileAdmin.search_fields,
        ProfileAdmin.list_select_related,
        ProfileAdmin.list_per_page,
        admin_model)
    cl.query_set = cl.query_set.filter(user=request.user)
    cl.get_results(request)
 
    context = {
        'title': cl.title,
        'is_popup': cl.is_popup,
        'cl': cl,
        'has_add_permission': admin_model.has_add_permission(request),
        'root_path': admin_model.admin_site.root_path,
        'app_label': _('Profile'),
    }
 
    return render_to_response('admin/change_list.html',
        context, context_instance=RequestContext(request))



