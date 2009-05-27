
# used by "dj-ip" to prime ipython environment with imports of  model classes 
# corresponding to all non-skipped tables 

from dybsite.offdb.generated import models as gm
from env.offline.dj import Import, Dump
exec(str(Import(gm)))
exec(str(Dump(gm)))


