#!/usr/bin/env python
"""
"""
import os, logging, shutil, tarfile, copy
from common import timing, seconds
log = logging.getLogger(__name__)


class Tar(object):
    def __init__(self, path, mode="gz"):
        self.path = path 
        self.mode = mode

    @timing
    def create(self, fromd, delete=True):
        """
        NB the below stores paths within the `tmp_offline_db` folder:: 
    
           t = Tar("/tmp/out/out.tar.gz")
           t.create("/tmp/out/tmp_offline_db")
           t.examine()

        ie::       
 
            <TarInfo './' at -0x48153114>
            <TarInfo 'SupernovaTrigger.MYD' at -0x48156b74>
            <TarInfo 'CalibPmtFineGainVld.frm' at -0x48156f94>
            <TarInfo 'HardwareID.MYD' at -0x48156a54>

        """
        log.info("creating %s from %s " %  (self.path, fromd) )
        tgz = tarfile.open(self.path, "w:%s" % self.mode )
        tgz.add(fromd, arcname="") 
        tgz.close() 
        if delete:
            shutil.rmtree(fromd)

    def examine(self):
        assert os.path.exists(self.path), "path %s does not exist " % self.path 
        log.info("examining %s " % (self.path) )
        tf = tarfile.open(self.path, "r:gz")
        for ti in tf:
            print ti.name
        print dir(ti)
        tf.close() 


    @timing
    def extract(self, extractdir, topleveldir=None, clobber=False):
        """
        :param extractdir:
        :param topleveldir: if specified only members within the topleveldir are extracted, protection against exploding tarballs
        :param clobber:

        hmm need to know the directories inside the tarfile
        """
        assert os.path.exists(self.path), "path %s does not exist " % self.path 
        if os.path.exists(extractdir):
            if not clobber:
                log.warn("extractdir %s exists already, use clobber option to delete it and proceed" % extractdir )        
                return
            else:
                pass
                #log.warn("deleting extractdir %s " % extractdir )  
                ## hmm no cant do that its the tld created by the extraction that need to delete, 

        log.info("extracting %s into %s " % (self.path,extractdir) )
        tf = tarfile.open(self.path, "r:gz")
        if topleveldir:
            members = filter(lambda ti:ti.name.split('/')[0] == topleveldir, tf)
        else:
            members = None

        wtf = TarFileWrapper(tf)
        wtf.extractall(extractdir, members) 
        tf.close() 


class TarFileWrapper(object):
    """
    Extractall only appears in 2.7 so back port from there into this wrapper from use from 2.3. 2.4, 2.5, 2.6  
    """
    def __init__(self, tf):
        self.tf = tf

    def extractall(self, path=".", members=None):
        """Extract all members from the archive to the current working
           directory and set owner, modification time and permissions on
           directories afterwards. `path' specifies a different directory
           to extract to. `members' is optional and must be a subset of the
           list returned by getmembers().
        """
        directories = []

        if members is None:
            members = self.tf

        for tarinfo in members:
            if tarinfo.isdir():
                # Extract directories with a safe mode.
                directories.append(tarinfo)
                tarinfo = copy.copy(tarinfo)
                tarinfo.mode = 0700
            self.tf.extract(tarinfo, path)

        # Reverse sort directories.
        directories.sort(lambda a, b: cmp(a.name, b.name))
        directories.reverse()

        # Set correct owner, mtime and filemode on directories.
        for tarinfo in directories:
            dirpath = os.path.join(path, tarinfo.name)
            try:
                self.tf.chown(tarinfo, dirpath)
                self.tf.utime(tarinfo, dirpath)
                self.tf.chmod(tarinfo, dirpath)
            except ExtractError, e:
                if self.tf.errorlevel > 1:
                    raise
                else:
                    self.tf._dbg(1, "tarfile: %s" % e)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    t = Tar("/var/scm/mysqlhotcopy/20130515_1249.tar.gz")
    t.examine()
    #t.extract("/tmp/out")

    #t = Tar("/tmp/out/out.tar.gz")
    #t.create("/tmp/out/tmp_offline_db")
    #t.examine()






