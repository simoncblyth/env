class SVNInfo(dict):
    """
        Access to svn info attributes for a working copy directory, usage
            from env.svn import SVNInfo
            si = SVNInfo("/Users/blyth/env")
            si['URL']
            
        The call form, avoids key errors in case of problems, like not working copy dir
            si('URL') == 'http://blah...'
            si('url') == None
            
        This access from anywhere requires :    
            * egglinking the env python package into sys.path into your targetted python 
              see env.bash/env-egglink
            * python package arrangement with __init__.py files in ~/env and ~/env/svn
            
        
    """
    def __init__(self, dir ):
        """
            Note potential for non english locale to mess this up 
        """
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
                    self[m.group('k')] = m.group('v')
        else:
            self['returncode'] = prc.returncode 
            self['out'] = out
            self['error'] = err

    def __call__(self, k , default=None ):
        return self.get(k, default )
        


if __name__=='__main__':
    si = SVNInfo('/Users/blyth/env')
    print si

