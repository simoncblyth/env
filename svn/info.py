
from env.structure import Persistent

class SVNInfo(Persistent):
    """
        Access to svn info attributes for a working copy directory, usage
            from env.svn import SVNInfo
            si = SVNInfo(dir="/Users/blyth/env")
            si['URL']
            
        The call form, avoids key errors in case of problems, like not working copy dir
            si('URL') == 'http://blah...'
            si('url') == None
            
        This access from anywhere requires :    
            * egglinking the env python package into sys.path into your targetted python 
              see env.bash/env-egglink
            * python package arrangement with __init__.py files in ~/env and ~/env/svn
            
        
    """
    def __init__(self, dir='/Users/blyth/env' ):
        self.info = {}
        self.parse(dir)
        
    def parse(self, dir):
        import subprocess
        prc = subprocess.Popen('svn info %s' % dir , shell=True , stdout=subprocess.PIPE , stderr=subprocess.PIPE  )
        out, err = prc.communicate()
        if prc.returncode == 0:
            import re
            lines = out.split("\n")
            att = re.compile('(?P<k>.*): (?P<v>.*)')
            for line in lines:
                m = att.match(line)
                if m:
                    self.info[m.group('k')] = m.group('v')
        else:
            self.info['returncode'] = prc.returncode 
            self.info['out'] = out
            self.info['error'] = err

    def __call__(self, k , default=None ):
        return self.info.get(k, default )
        
    def __repr__(self):
        import pprint
        return pprint.pformat(self.info)


if __name__=='__main__':
    from env.svn import SVNInfo
    e = SVNInfo(dir='/Users/blyth/env')
    w = SVNInfo(dir='/Users/blyth/workflow') 
    print e
    print w

