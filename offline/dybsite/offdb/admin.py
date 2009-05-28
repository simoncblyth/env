"""
  http://docs.djangoproject.com/en/dev/intro/tutorial02/

"""

from django.contrib import admin

from env.offline.dj import Import, Admin
from dybsite.offdb.generated import models as gm
exec(str(Import(gm)))
exec(str(Admin(gm)))


if __name__=='__main__':
    print Import(gm)
    print Admin(gm)

