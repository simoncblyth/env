from django.db import models

class Job(models.Model):
    name      = models.CharField(max_length=50)

class JobField(models.Model):
    container = models.ForeignKey(Job, db_index=True)
    key       = models.CharField(max_length=240, db_index=True)
    value     = models.CharField(max_length=240, db_index=True)



