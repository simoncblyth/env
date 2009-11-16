# http://ask.github.com/celery/introduction.html
# http://ask.github.com/celery/tutorials/clickcounter.html

from celery.task import PeriodicTask
from celery.registry import tasks
from abtruninfo.messaging import process_abtruninfo
from datetime import timedelta

class ProcessRunTask(PeriodicTask):
    run_every = timedelta(minutes=1)
    def run(self, **kwargs):
        process_abtruninfo()

tasks.register(ProcessRunTask)





