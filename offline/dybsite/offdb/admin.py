"""
  http://docs.djangoproject.com/en/dev/intro/tutorial02/

"""

from django.contrib import admin

from env.offline.dj import Import, Register
from dybsite.offdb.generated import models as gm
exec(str(Import(gm)))
exec(str(Register(gm)))

#from dybsite.offdb.models import PSimPmtSpec
#
#class PSimPmtSpecAdmin(admin.ModelAdmin):
#    fields = ['seqno', 'row_counter', 'pmtsite', 'pmtad', 'pmtring', 'pmtcolumn', 'pmtgain', 'pmtgfwhm', 'pmttoffset', 'pmttspread', 'pmteffic', 'pmtprepulse', 'pmtafterpulse', 'pmtdarkrate']
#
#admin.site.register(PSimPmtSpec, PSimPmtSpecAdmin)



if __name__=='__main__':
    print Import(gm)
    print Register(gm)

