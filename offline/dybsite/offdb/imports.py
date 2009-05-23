
# used by "dj-ip" to prime ipython environment with imports of  model classes 
# corresponding to all non-skipped tables 

from env.offline.dybsite.offdb.generated import models as genmodels
from env.offline.dj import ExamineGenModels
egm = ExamineGenModels(genmodels)
print "importing %s from %s " % ( egm.proxynames() , egm.modulename )
exec(egm.import_all())
print "creating dump_all(): function "
exec(egm.dump_all())


