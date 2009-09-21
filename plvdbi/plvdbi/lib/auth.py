
from pylons.templating import render_genshi as render

class FlashDummy:
    def render(self,*args):
        return '<div></div>'

def render_signin():  
    """
       Attempting to access the live vdbi_app can cause threading issues... so 
       try to mock-up the vdbi template context statically.  
    """      
    from vdbi.rum.widgets import DEFAULT_WIDGETS as widgets
    extra_vars = { 
       'widgets':widgets,
       'master_template':"master.html",
       'resources':[],
       'url_for':lambda x:x,
       'flash':FlashDummy(),
    } 
    return render("dbilogin.html", extra_vars=extra_vars )

