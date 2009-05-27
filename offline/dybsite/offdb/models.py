
## dynamic proxy wrapping to stay current with db schema 

from dybsite.offdb.generated import models as gm
from env.offline.dj import Proxy
exec str(Proxy(gm))
