
from rum import app

from tw.api import Widget
from tw.api import Widget, Link, lazystring as _

## pointing to static 
modname = "vdbi.tw.rum"  



class DbiCollectionLinks(Widget):
    params = ['field', 'icons', 'actions', 'show_items']
    
    actions = [
        ('index', _('Show all')),
        ('new', _('New')),
        ]
    show_items=False
    template = "genshi:tw.rum.templates.collection_links"
    icons = {
        'show': Link(modname=modname, filename="static/pencil_go.png"),
        }

    def update_params(self, d):
        super(DbiCollectionLinks, self).update_params(d)
        actions=self.actions
        if d.field.read_only:
            actions=[(link, label) for (link, label) in actions if link!="new"]
        def url_for_item(i):
            return app.url_for(obj=i, _memory=False)
        d.url_for_item=url_for_item
        if d.show_items:
            actions=[(a,desc) for (a,desc) in actions if a!="index"]
            d.items=getattr(d.value, d.field.name)
        d.links = []
        for action, title in actions:
            url = app.url_for(parent_obj=d.value, action=action,
                              resource=d.field.other,
                              remote_name=d.field.name,
                              _memory=False)
            d.links.append((url, title, title))

