from django.db import models
from datetime import datetime

operator_ = {
  'choices':(
              ( 0 , 'Jimmy Ngai'),
              ( 1 , 'Talent'),
              ( 2 , 'Soap'),
              ( 3,  'KamBiu'),
            ),
   'default':0,
}

source_ = {
 'choices':(  
             ( 0 , 'None'),
             ( 1 , 'Co-60'),
             ( 2,  'Co-60 (Cal Box)'),
           ),
  'default':0,
}

trigger_ = {
  'choices':(
              (0, 'Default'),
              (1, '16 out of 16'),
           ),
  'default':0,
}


run_admin_display = ('number','start','stop','events','operator','tkoffset','source','pmtgain','trigger','temperature','humidity','comment','frontendhost','frontendname','created')
run_admin_filter  = ('operator','source','trigger','frontendhost','frontendname')

class Run(models.Model):
    number    = models.PositiveIntegerField( primary_key=True)
    start     = models.DateTimeField()
    stop      = models.DateTimeField()
    events    = models.PositiveIntegerField()
    operator  = models.PositiveSmallIntegerField( **operator_ )
    tkoffset  = models.PositiveIntegerField()
    source    = models.PositiveSmallIntegerField( **source_ )
    pmtgain   = models.FloatField(),
    trigger   = models.PositiveSmallIntegerField( **trigger_ )
    temperature = models.FloatField(),
    humidity    = models.FloatField(),
    comment     = models.CharField(max_length=200)
    frontendhost  = models.CharField(max_length=40)
    frontendname  = models.CharField(max_length=40)

    created   = models.DateTimeField('created', default=datetime.now)
    class Meta:
        ordering = ['-created']
        get_latest_by = 'created'



