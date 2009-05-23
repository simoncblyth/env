
import inspect

def skip( name ):return name.startswith('Auth') or name.startswith('Django')
def proxy_name_( cls ):return 'P' + cls._meta.db_table

class ExamineGenModels:
    """
           from env.offline.dybsite.offdb.generated import models as genmodels
           from env.offline.dj import ExamineGenModels
           egm = ExamineGenModels(genmodels)
           print egm.names()
    """
    def __init__(self, module):
        assert module.__name__.endswith('generated.models'), module.__name__
        self.module = module
        self.modulename = module.__name__.replace('generated.models','models')
    def names(self): 
        return [name for name,cls in inspect.getmembers( self.module , inspect.isclass ) if not(skip(name))]
    def proxynames(self): 
        return [proxy_name_(cls) for name,cls in inspect.getmembers( self.module , inspect.isclass ) if not(skip(name))]

    def import_(self, proxy_name ):
        modulename = self.modulename
        return "from %(modulename)s import %(proxy_name)s" % locals()  
    def import_all(self):
        return "\n".join( [self.import_(name) for name in self.proxynames()]) 

    def dump_(self, proxy_name ):
        return "    for o in %(proxy_name)s.objects.all():print o" % locals()  
    def dump_all(self):
        return "\n".join( ["def dump_all():"] + [self.dump_(name) for name in self.proxynames()]) 
  



tmpl = r"""
from %(modulename)s import %(name)s
class %(proxy_name)s(%(name)s):
    class Meta:
        proxy = True
    def __unicode__(self):
        return "<%(name)s %(ffmt)s > " %(perc)s  ( %(sfmt)s ) 
"""


class ProxyWrap:
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

 
    def __init__(self, module):self.module = module
    def __repr__(self):
        modulename = self.module.__name__
        codegen = ""
        import inspect
        for name,cls in inspect.getmembers( self.module , inspect.isclass ):
            if not(skip(name)):
                cols = [field.name for field, modl in cls._meta.get_fields_with_model()]
                proxy_name = proxy_name_(cls)
                ffmt = "%s " * len(cols)
                sfmt = ",".join( [ "self.%s  " % col for col in cols ])
                perc = '%'   
                codegen += tmpl % locals()
            pass
        return codegen


if __name__=='__main__':
    from env.offline.dybsite.offdb.generated import models as genmodels
    from env.offline.dj import ExamineGenModels
    egm = ExamineGenModels(genmodels)
    print egm.import_all()
    print egm.dump_all()


