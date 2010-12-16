
from converter import restwriter, convert_file
import os


base = os.environ.get('DYB')
texdir = os.path.join( base ,  "NuWa-trunk/dybgaudi/Documentation/OfflineUserManual/tex" ) 
texdir = os.path.join( texdir, 'database' )
outdir = '/tmp/out'

for i in os.listdir(texdir):
    name, ext = os.path.splitext( i )
    if ext == '.tex':
        inpath = os.path.join( texdir, i )
        outpath = os.path.join( outdir, "%s.rst" % name )
        print "%s --> %s " % ( inpath , outpath )   
        convert_file( inpath, outpath )











"""
class IncludeRewrite:
    def get(self, a, b=None):
       if os.path.exists(os.path.join(source, a + '.tex')):
          return a + '.rst'
      return a

restwriter.includes_mapping = IncludeRewrite()

for infile, outfile in :
   convert_file(infile, outfile)
"""
