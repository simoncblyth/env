#!/usr/bin/env python
"""
"""
import os, logging, shutil, tarfile, copy, re
from common import timing, seconds, scp
from datetime import datetime
log = logging.getLogger(__name__)


class Tar(object):
    def __init__(self, path, toplevelname="", mode="gz", remoteprefix="", remotenode="C" , confirm=True ):
        """
        :param path: to the tarball to be created, extracted or examined
        :param toplevelname: relative to the sourcedir or extractdir, 

        if a `toplevelname` is specified only members within that directory 
        are tarballed or extracted

        Hmm embedding the toplevel name withing the tarball, is not so flexible 
        when want to test an extracted mysql DB tarball, would be more convenient to just flat 
        archive the files. 
        """
        assert len(toplevelname) > 1, "as safety measure a non-blank toplevelname is required" 
        self.path = path 
        self.toplevelname = toplevelname
        self.mode = mode
        if len(remoteprefix)>0:
            remotepath = os.path.join(remoteprefix, path[1:])   # have to get rid of path leading slash for the join
        else:
            remotepath = path
        pass
        self.remotepath = remotepath
        self.remotenode = remotenode
        self.confirm = confirm 


    def __repr__(self):
        return self.__class__.__name__ + " %s %s %s " % ( self.path, self.toplevelname, self.mode )

    def examine(self):
        assert os.path.exists(self.path), "path %s does not exist " % self.path 
        log.info("examining %s " % (self.path) )
        tf = tarfile.open(self.path, "r:gz")
        for ti in tf:
            print ti.name
        print dir(ti)
        tf.close() 
    examine = timing(examine)

    def archive(self, sourcedir, deleteafter=False):
        """
        :param sourcedir: directory containing the `toplevelname` which will be the root of the archive 
        :param deleteafter:

        Create the archive and examine::

           t = Tar("/var/dbbackup/mysqlhotcopy/belle7.nuu.edu.tw/tmp_offline_db/20130515_1941.tar.gz", toplevelname="tmp_offline_db")
           t.archive("/var/dbbackup/mysqlhotcopy/belle7.nuu.edu.tw/tmp_offline_db/20130515_1941")
           t.examine()

        Examine what is in the archive:: 
    
           t = Tar("/var/dbbackup/mysqlhotcopy/belle7.nuu.edu.tw/tmp_offline_db/20130515_1941.tar.gz", toplevelname="tmp_offline_db")
           t.examine()

        Under toplevelname `tmp_offline_db` within the archive::       

            tmp_offline_db/
            tmp_offline_db/SupernovaTrigger.MYD
            tmp_offline_db/CalibPmtFineGainVld.frm
            tmp_offline_db/HardwareID.MYD
            ...

        To reproduce the layout on another node would then need::

           t = Tar("/var/dbbackup/mysqlhotcopy/belle7.nuu.edu.tw/tmp_offline_db/20130515_1941.tar.gz", toplevelname="tmp_offline_db")
           t.extract("/tmp")  # creates /tmp/tmp_offline_db 

        """
        src = os.path.join(sourcedir, self.toplevelname) 
        assert len(self.toplevelname) > 3 , "sanity check for toplevelname %s fails" % self.toplevelname
        log.info("creating %s from %s " %  (self.path, src) )
        assert os.path.exists(src) and os.path.isdir(src), "src directory %s does not exist " % src
        tgz = tarfile.open(self.path, "w:%s" % self.mode )
        tgz.add(src, arcname=self.toplevelname) 
        tgz.close() 

        datedfolder_ptn = re.compile("^\d{8}_\d{4}$") # eg 20130515_1941
        if deleteafter:
            leaf = sourcedir.split("/")[-1]
            if not datedfolder_ptn.match(leaf):
                log.warn("NOT deleting sourcedir %s with leaf %s as the leaf is not a dated folder " % ( sourcedir, leaf ))
            else:
                log.info("deleting sourcedir %s with leaf %s as the leaf is a dated folder " % ( sourcedir, leaf ))
                if self.confirm:
                    confirm = raw_input("enter \"YES\" to confirm deletion of sourcedir %s :" % sourcedir )
                else:
                    confirm = "YES"
                pass 
                if confirm == "YES":
                    shutil.rmtree(sourcedir)
                else:
                    log.info("skipping deletion of %s " % sourcedir ) 
                pass
        else:
            log.warn("not deleteing after")
    archive = timing(archive)

    def extract(self, extractdir, moveaside=False):
        """
        :param extractdir: 
        :param moveaside:
        """
        assert os.path.exists(self.path), "path %s does not exist " % self.path 
        tgt = os.path.join(extractdir, self.toplevelname) 
        if os.path.exists(tgt):
            if not moveaside:
                log.warn("tgt dir %s exists already, ABORTING EXTRACTION, use --moveaside option to rename it and proceed" % tgt )        
                return
            else:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                aside = tgt + "_" + stamp
                log.warn("moving aside pre-existing tgt dir %s to %s " % (tgt, aside) )  
                assert not os.path.exists(aside), (aside, "huh the aside dir exists already ")
                os.rename(tgt, aside)

        assert not os.path.exists(tgt), "huh should not exist at this point "
        log.info("extracting %s with toplevelname %s into extractdir %s " % (self.path,self.toplevelname, extractdir) )
        tf = tarfile.open(self.path, "r:gz")
        members = tf.getmembers()
        select = filter(lambda tinfo:tinfo.name.split('/')[0] == self.toplevelname, members)
        assert len(members) == len(select), (len(members), len(select), "extraction filtering misses some members not beneath toplevelname %s " % self.toplevelname ) 
        pass
        wtf = TarFileWrapper(tf)
        wtf.extractall(extractdir, select) 
        tf.close() 
    extract = timing(extract)

    def transfer(self):
        """
        """
        assert os.path.exists(self.path), "path %s does not exist " % self.path 
        scp( self.path, self.remotepath, self.remotenode )
    transfer = timing(transfer)
        

 
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

    tgz = "/var/dbbackup/mysqlhotcopy/belle7.nuu.edu.tw/tmp_offline_db/20130515_1941.tar.gz"
    t = Tar(tgz, toplevelname="tmp_offline_db")
    t.examine()
    #t.extract("/tmp/out")
    #t.examine()

    log.info(seconds)




