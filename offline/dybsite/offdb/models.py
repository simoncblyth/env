
## dynamic proxy wrapping to stay current with db schema 

from env.offline.dybsite.offdb.generated import models as genmodels
from env.offline.dj import ProxyWrap
exec str(ProxyWrap(genmodels))
