"""

   Introspects the "inspectdb" generated models (excluding Auth* Django* tables)
   to determine the classes corresponding to tables and their attributes corresponding 
   to the fields.

   On updating the database tables use "dj-models" to regenerate ... 

   On importing :
         dybsite.offdb.models 
         dybsite.offdb.imports
         dybsite.offdb.admin

   This module manages code generation against the genrated models
   allowing to do generic manipulations to set up the admin site etc..

   Although using code generation is regarded as somewehat nasty    
   in this case it is a lot simpler than alternatives such as "new.classobj"
   or managing lots of temporary files.

        Import
             import all the proxy models
        Dump
             excercise the python api to dump the objects
        Proxy
             create models that proxy the generated models
             in order to avoid touching the generated models
        Register
              with the admin interface 


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



class VldProxy(dict):
    tmpl = r"""
from %(modulename)s import %(name)s
class %(proxy)s(%(name)s):
    class Meta:
        proxy = True
        verbose_name = "%(proxy)s"
    def __unicode__(self):
        return "%(proxy)s:%(ffmt)s" %(perc)s  ( %(sfmt)s ) 

%(name)s.simflag = lambda self:e('SimFlag',self.simmask )
%(name)s.site    = lambda self:e('Site',   self.sitemask )
%(name)s.__unicode__ = lambda self:"%(name)s:%(ffmt)s" %(perc)s ( %(sfmt)s )

databrowse.site.register(%(proxy)s)
"""
    def __init__(self, *a):
        for d in a:self.update(d)
    def __repr__(self):
       return self.__class__.tmpl % self  


class MainProxy(dict):
    tmpl = r"""
from %(modulename)s import %(name)s
class %(proxy)s(%(name)s):
    class Meta:
        proxy = True
        verbose_name = "%(proxy)s"
    def __unicode__(self):
        return "<%(name)s %(ffmt)s > " %(perc)s  ( %(sfmt)s ) 

%(name)s.simflag = lambda self:e('SimFlag',self.seqno.simmask )
%(name)s.site    = lambda self:e('Site',   self.seqno.sitemask )

databrowse.site.register(%(proxy)s)
"""
    def __init__(self, *a):
        for d in a:self.update(d)
    def __repr__(self):
        return self.__class__.tmpl % self  


class Models:
    tmpl = r"""
from django.contrib import databrowse
from env.offline.enum import Enum
e = Enum()

%(codegen)s

"""
    def __init__(self, module):
        self.dj = Dj(module)
    def __repr__(self):
        codegen = ""
        for name,cls in self.dj.items():
            proxy = proxy_name_(cls)
            cols = [field.name for field, modl in cls._meta.get_fields_with_model()]
            perc = '%'   
            if proxy.endswith('Vld'):
                ffmt = "%s"
                sfmt = "self.seqno"
                p = VldProxy( self.dj.ctx ,  locals() )
            else:
                ffmt = "%s " * len(cols)
                sfmt = ",".join( [ "self.%s  " % col for col in cols ])
                p = MainProxy( self.dj.ctx , locals()  )
            codegen += str(p)
        return self.dj.filltmpl( self.__class__.tmpl , locals() )




class Admin:
    tmpl = r"""

class %(proxy)sAdmin(admin.ModelAdmin):
    list_display = %(list_display)s
    list_filter = %(list_filter)s
    %(date_hierarchy)s

admin.site.register(%(proxy)s, %(proxy)sAdmin)
"""
    ## generalize to a non PK integer ?
    filters = ['pmtsite','pmtad','pmtring','pmtcolumn']  
    def __init__(self, module):self.dj = Dj(module)
    def __repr__(self):
        """
           display order is arranged to put the pk first 
        """
        codegen = ""
        for name, cls in self.dj.items():
            proxy = proxy_name_(cls)
            fields = [field.name for field, modl in cls._meta.get_fields_with_model() if field != cls._meta.pk  ]
            if proxy.endswith("Vld"):
                 date_hierarchy = "date_hierarchy = 'insertdate'" 
                 fields += ['site','simflag']
            else:
                 fields += ['site','simflag']
                 date_hierarchy = "pass"

            list_display = str([cls._meta.pk.name] + fields )
            list_filter  = str([f for f in fields if f in self.__class__.filters])
            codegen += self.dj.filltmpl( self.__class__.tmpl , locals() )
        return codegen







if __name__=='__main__':
    from dybsite.offdb.generated import models as gm
    print Import(gm)
    print Dump(gm)
    print Models(gm)
    print Admin(gm)

    from env.offline.dj import Dj
    dj = Dj(gm)
    print dj


