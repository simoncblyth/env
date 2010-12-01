
from  distutils.sysconfig import get_config_vars 
for k,v in get_config_vars().items():
    if "2.5" in str(v):
        print k,v  


