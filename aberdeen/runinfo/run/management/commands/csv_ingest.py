"""
    http://andrewwilkinson.wordpress.com/2009/03/06/creating-django-management-commands/
    http://docs.djangoproject.com/en/dev/howto/custom-management-commands/
"""

from django.core.management.base import NoArgsCommand, BaseCommand, CommandError
from django.core.management import call_command
from django.conf import settings

from runinfo.run.models import *
from optparse import make_option
import os

def decode_row( row ):
    number = int(row[0].lstrip('#'))
    start = datetime.strptime( row[1] , "%c")
    stop = datetime.strptime( row[2] , "%c")
    try:
        events = int(row[3].rstrip('k'))
    except ValueError:
        events = 0
    operator = operator_.for_name( row[4] )
    tkoffset = tkoffset_.for_name( row[5] )
    source = source_.for_name( row[6] )

    if row[7] == 'Default':
        pmtgain = 0.
    else:
        pmtgain = float(row[7])
    trigger = trigger_.for_name( row[8] )
    
    if row[9].find("nan")>-1:
        temperature = 0.
    else:
        temperature = float(row[9])
    humidity = float(row[10])
    comment = row[11]
    frontendhost = row[12]
    frontendname = row[13]

    d = locals()
    del d['row']
    return Run( **d )


def csv_get( url ):
    nam = os.path.basename(url)
    if not(os.path.exists(nam)):
        print os.popen("curl -O \"%s\" " % url ).read()
    return os.path.join( "." , nam )

def csv_decode( path ):
    import csv
    fp = open( path , "rb" )
    reader = csv.reader( fp )
    for row in reader:
        r = decode_row( row )
        print r  
        r.save()

def csv_ingest( **options ):
     print "ingesting ... %s " % repr(options)
     url = options.get('url',None)
     if not(url):url = "http://dayabay8core.no-ip.org:8081/CS/runlogfile"
     path = csv_get( url )
     if os.path.exists(path):
         csv_decode( path )
     else:
         print "no such path %s " % path

      
class Command(BaseCommand):
    option_list = BaseCommand.option_list + (
        make_option('--url', '-p', dest='url', help='URL of csv file '),
    )
    help = 'Ingest from csv file '

    def handle(self, **options):
        csv_ingest(**options)


