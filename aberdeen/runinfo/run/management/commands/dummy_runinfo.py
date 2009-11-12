from django.core.management.base import NoArgsCommand, BaseCommand, CommandError
from django.core.management import call_command
from django.conf import settings

from runinfo.run.messaging import send_dummy_runinfo
from runinfo.run.models import *
from optparse import make_option
import os

      
class Command(BaseCommand):
    option_list = BaseCommand.option_list + (
        make_option('--url', '-p', dest='url', help='URL of csv file '),
    )
    help = 'Ingest from csv file '

    def handle(self, **options):
        dummy = {'hello':"world" } 
        send_dummy_runinfo(dummy)


