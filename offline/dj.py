"""
         Creates Django proxy models for django 
         models introspected from the database tables, excluding 
         standard tables with names Auth* Django*

         The idea is to avoid touching the genmodels and 
         do the generic hookups that can be done

         Usage :

            1) generate models from database with 
                   dj-manage inspectdb > genmodels.py

            2) place a models.py containing something like:

                 from env.offline.dybsite.offdb import genmodels
                 from env.offline.dj import ProxyWrap
                 exec str(ProxyWrap(genmodels))

        This is somewhat nasty as it uses code generation  
        but this is a lot simpler than using new.classobj
"""

def skip( name ):return name.startswith('Auth') or name.startswith('Django')
def proxy_name_( cls ):return 'P' + cls._meta.db_table

class Dj(dict):
    def __init__(self, module ):
        assert module.__name__.endswith('generated.models'), module.__name__
        import inspect
        for name,cls in inspect.getmembers( module , inspect.isclass ):
             if not(skip(name)):
                 self[name] = cls
        ctx = {}
        ctx['modulename'] = module.__name__
        ctx['proxymodulename'] = module.__name__.replace('generated.models','models')
        self.ctx = ctx
        self.module = module
        pass
    def names(self):return self.keys()
    def proxynames(self):return [proxy_name_(cls) for cls in self.values()]
    def filltmpl(self, tmpl, extra ):return tmpl  % dict( self.ctx , **extra ) 



class Import:
    tmpl = r"""
from %(proxymodulename)s import %(proxy)s
"""
    def __init__(self, module):self.dj = Dj(module)
    def __repr__(self):
        return "\n".join( [ self.dj.filltmpl( self.__class__.tmpl , locals() ) for proxy in self.dj.proxynames() ] )


class Dump:
    head = r"""
def dump_all():
"""
    tmpl = r"""
    for o in %(proxy)s.objects.all():print o
"""
    def __init__(self, module):self.dj = Dj(module)
    def __repr__(self):
        return "\n".join( [ self.__class__.head ] +  [ self.dj.filltmpl( self.__class__.tmpl , locals() ) for proxy in self.dj.proxynames() ] )


class Proxy:
    tmpl = r"""
from %(modulename)s import %(name)s
class %(proxy_name)s(%(name)s):
    class Meta:
        proxy = True
    def __unicode__(self):
        return "<%(name)s %(ffmt)s > " %(perc)s  ( %(sfmt)s ) 
"""
    def __init__(self, module):
        self.dj = Dj(module)
    def __repr__(self):
        codegen = ""
        for name,cls in self.dj.items():
            cols = [field.name for field, modl in cls._meta.get_fields_with_model()]
            proxy_name = proxy_name_(cls)
            ffmt = "%s " * len(cols)
            sfmt = ",".join( [ "self.%s  " % col for col in cols ])
            perc = '%'   
            codegen += self.dj.filltmpl( self.__class__.tmpl , locals() )
        return codegen


class Register:
    tmpl = r"""
class %(proxy_name)sAdmin(admin.ModelAdmin):
    fields = %(fields)s

admin.site.register(%(proxy_name)s, %(proxy_name)sAdmin)
"""
    def __init__(self, module):self.dj = Dj(module)
    def __repr__(self):
        codegen = ""
        for name, cls in self.dj.items():
            proxy_name = proxy_name_(cls)
            fields = str([field.name for field, modl in cls._meta.get_fields_with_model()])
            codegen += self.dj.filltmpl( self.__class__.tmpl , locals() )
        return codegen


if __name__=='__main__':
    from dybsite.offdb.generated import models as gm
    print Import(gm)
    print Dump(gm)
    print Proxy(gm)
    print Register(gm)

    from env.offline.dj import Dj
    dj = Dj(gm)
    print dj


