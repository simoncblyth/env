from env.offline.dbconf import DBConf
DATABASES = dict([ (k,DBConf(k,verbose=False).django) for k in "default prior client".split() ] )  

print DATABASES


