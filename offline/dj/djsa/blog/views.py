"""
   Attempt to get an out-of-admin changelist to work
   with sqlalchemy underpinnings

"""

## http://blog.stiod.com/2008/12/16/reaproveitando-a-inteface-do-admin-do-django/

from django.shortcuts import render_to_response
from django.template import RequestContext
from django.utils.text import ugettext_lazy as _

from djsa.blog.models import Post
from djsa.blog.admin  import PostAdmin

from django.contrib import admin
from django.contrib.admin.views.main import ChangeList
 
def post_index(request):
    admin_model = admin.site._registry[Post]
    admin_model.admin_site.root_path = request.path

    cl = ChangeList(
        request,
        Post,
        PostAdmin.list_display,
        PostAdmin.list_display_links,
        PostAdmin.list_filter,
        PostAdmin.date_hierarchy,
        PostAdmin.search_fields,
        PostAdmin.list_select_related,
        PostAdmin.list_per_page,
        PostAdmin.list_editable,
        admin_model)
    #cl.query_set = cl.query_set.filter(user=request.user)
    cl.get_results(request)
    cl.formset = None

    context = {
        'title': cl.title,
        'is_popup': cl.is_popup,
        'cl': cl,
        'has_add_permission': True, #admin_model.has_add_permission(request),
        'root_path': admin_model.admin_site.root_path,
        'app_label': _('Post'),
    }
 
    return render_to_response('admin/change_list.html',
        context, context_instance=RequestContext(request))



