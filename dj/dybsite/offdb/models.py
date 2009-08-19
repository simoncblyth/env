
## dynamic proxy wrapping to stay current with db schema 

from dybsite.offdb.generated import models as gm
from env.dj import Models 
exec str(Models(gm))
