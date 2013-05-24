#!/usr/bin/env python
"""
"""
import os, logging, shutil, tarfile, copy, re, pickle
from common import timing, seconds, scp
import fsutils
from datetime import datetime
log = logging.getLogger(__name__)


class Tar(object):
    def __init__(self, path, toplevelname="", mode="gz", remoteprefix="", remotenode="C" , confirm=True, moveaside=True, ALLOWCLOBBER=False):
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
        self.moveaside = moveaside
        self.ALLOWCLOBBER = ALLOWCLOBBER
        self.names = None
        self.prefix = None
        self.flattop = None 

    def __repr__(self):
        return self.__class__.__name__ + " %s %s %s " % ( self.path, self.toplevelname, self.mode )

    def members_(self):
        """
        Caches the members list from tarballs into a sidecar `.pc` file
        to avoid a 70s wait to access the members of a compressed tarball

        http://userprimary.net/posts/2007/11/18/ctime-in-unix-means-last-change-time-not-create-time/

             ctime means change time
        """
        path = self.path
        mtime_ = lambda _:os.path.getmtime(_)
        ctime_ = lambda _:os.path.getctime(_)
        pc = "%s.pc" % path
        members = None
        if os.path.exists(pc):
            if ctime_(pc) > ctime_(path):
                log.warn("load pickled members file %s " % pc )
                members = pickle.load(file(pc,"r")) 
            else:
                log.warn("pickled members exists but is outdated")
            pass
        pass 
        if not members:
            tf = tarfile.open(path, "r:gz")
            members = tf.getmembers() 
            pickle.dump( members, file(pc,"w")) 
            log.info("saving pickled members file: %s " % pc)
            tf.close() 
        pass
        return members

    def names_(self):
        members = self.members_()
        names = map(lambda ti:ti.name, members)
        return names

    def examine(self):
        assert os.path.exists(self.path), "path %s does not exist " % self.path 
        log.info("examining %s " % (self.path) )
        names = self.names_()
        prefix = os.path.commonprefix(names)
        flattop = prefix == ""

        log.info("archive contains %s items with commonprefix \"%s\" flattop %s " % ( len(names), prefix, flattop  ))
        log.debug("\n".join(names))

        self.names = names
        self.prefix = prefix
        self.flattop = flattop 
    examine = timing(examine)

    def archive(self, sourcedir, deleteafter=False, flattop=False):
        """
        :param sourcedir: directory containing the `toplevelname` which will be the root of the archive 
        :param deleteafter:
        :param flattop:

        Create the archive and examine::

           t = Tar("/var/dbbackup/mysqlhotcopy/belle7.nuu.edu.tw/tmp_offline_db/20130515_1941.tar.gz", toplevelname="tmp_offline_db")
           t.archive("/var/dbbackup/mysqlhotcopy/belle7.nuu.edu.tw/tmp_offline_db/20130515_1941")
           t.examine()

        Examine what is in the archive:: 
    
           t = Tar("/var/dbbackup/mysqlhotcopy/belle7.nuu.edu.tw/tmp_offline_db/20130515_1941.tar.gz", toplevelname="tmp_offline_db")
           t.examine()

        Under toplevelname `tmp_offline_db` within the archive when `flattop=False`::

            tmp_offline_db/
            tmp_offline_db/SupernovaTrigger.MYD
            tmp_offline_db/CalibPmtFineGainVld.frm
            tmp_offline_db/HardwareID.MYD
            ...

        Under toplevelname `tmp_offline_db` within the archive when `flattop=True`::

            SupernovaTrigger.MYD
            CalibPmtFineGainVld.frm
            HardwareID.MYD
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
        if flattop:
            arcname = ""
        else:
            arcname = self.toplevelname 
        pass
        tgz.add(src, arcname=arcname) 
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


    def moveaside(self, target, dryrun=False):
        assert os.path.exists(target), target
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        aside = target + "_" + stamp
        msg = "moving aside pre-existing tgt dir %s to %s " % (target, aside) 
        assert not os.path.exists(aside), (aside, "huh the aside dir exists already ")
        if dryrun:
            log.info("dryrun : " + msg )
            return
        pass
        log.info(msg) 
        os.rename(target, aside)


    def _folder_extract(self, containerdir, toplevelname, dryrun=False):
        """  
        :param containerdir:
        :param toplevelname:
        :param dryrun:

        Folder extraction takes all paths from the archive that are within 
        a particular toplevelname within the archive and places then 
        within the `containerdir/` folder. By virtue of the `toplevelname` 
        paths within the archive the result will be 
        `containerdir/toplevelname` folder in the filesystem.

        This approach has the advantage of a non-exploding tarball, but is 
        inconvenient for renaming. 

        The `toplevelname` dir will be created by the extraction.

        """
        assert self.flattop is False , "_folder_extract requires non-flattop archive "
        assert self.toplevelname == toplevelname ,"_folder_extract requires default toplevelname %s %s " % (self.toplevelname, toplevelname)
        tf = tarfile.open(self.path, "r:gz")
        wtf = TarFileWrapper(tf)
        members = tf.getmembers()
        select_ = lambda tinfo:tinfo.name.split('/')[0] == toplevelname
        select = filter(select_, members)
        assert len(members) == len(select), (len(members), len(select), "extraction filtering misses some members, toplevelname %s " % (toplevelname) ) 
        target = os.path.join(containerdir, toplevelname)

        if os.path.exists(target) and self.moveaside:
            self.moveaside(target, dryrun=dryrun)
        pass

        msg = "_folder_extract into containerdir %s for %s members with toplevelname %s  " % ( containerdir, len(members), toplevelname )
        if dryrun:
            log.info("dryrun: " + msg )
        else:
            log.info(msg)
            assert not os.path.exists(target), "target dir %s exists already, ABORTING EXTRACTION, use --moveaside option to rename it " % target
            wtf.extractall(containerdir, members) 
        pass
        tf.close() 


    def _check_clobber(self, target, members ):
        """
        :param target: directory in which the members are to be extracted
        :param members: from the tarfile
        :return: list of paths that would be clobberd by the extraction
        """
        clobber = []
        fmt = "%-110s :  %s " 
        for member in members:
            name = member.name
            path = os.path.join(target, name)
            if os.path.exists(path):
                if name == './':
                    log.warn(fmt % (name, "SKIP TOPDIR" ))
                else:
                    clobber.append(name)
                    log.warn(fmt % (name, "**CLOBBER**" ))
            else:
                log.debug(fmt % (name, "" ))
            pass
        pass
        return clobber


    def _flat_extract(self, containerdir, toplevelname, dryrun=False):
        """
        :param containerdir:
        :param toplevelname:
        :param dryrun:

        Flat extraction takes all paths from the archive and places
        them within the `containerdir/toplevelname` folder 

        The `toplevelname` dir must be created before the extraction.

        """
        assert self.flattop is True , "_flat_extract requires flattop archive "
        log.info("_flat_extract opening tarfile %s " % self.path )
        tf = tarfile.open(self.path, "r:gz")
        wtf = TarFileWrapper(tf)
        members = tf.getmembers()
        target = os.path.join(containerdir, toplevelname)

        clobber = self._check_clobber( target, members )
        if len(clobber) > 0:
            if not self.ALLOWCLOBBER:
                log.warn("extraction would clobber %s existing paths, need `--ALLOWCLOBBER` option to do this : %s " % ( len(clobber), "\n".join(clobber) ))   
            else:
                low.warn("proceeding to clobber %s existing paths curtesy of `--ALLOWCLOBBER` option : %s " %  ( len(clobber), "\n".join(clobber) )) 
        else:
            log.info("extraction into target %s does not clobber any existing paths " % target )   


        msg = "_flat_extract into target %s for %s members with toplevelname %s " % ( target, len(members),toplevelname )
        if dryrun:
            log.info("dryrun: " + msg )
        else:
            log.info(msg)
            if not self.ALLOWCLOBBER:
                assert not os.path.exists(target), "target dir %s exists already, ABORTING EXTRACTION use --rename newname " % target 
            wtf.extractall(target, members) 
            pass 
            log.info( os.popen("ls -l %(target)s " % locals()).read() )
        pass
        tf.close() 



    def extract(self, containerdir, toplevelname=None, dryrun=False):
        """
        :param containerdir: folder within which the toplevelname dir resides
        :param toplevelname: default of None corresponds to original db name
        :param dryrun:

        The actual extraction method depends on the type of archive detected:

        #. `_flat_extract` for a flattop aka exploding archive 
        #. `_folder_extract` for a folder top archive  

        Flat extraction has the advantage of easy renaming  
        """
        if toplevelname is None:
            toplevelname = self.toplevelname
        pass
        assert os.path.exists(self.path), "path %s does not exist " % self.path 
        assert os.path.exists(containerdir), "containerdir %s does not exist" % containerdir
        assert not self.flattop is None, "ABORT must `examine` before can `extract` "

        if self.flattop:
             self._flat_extract(containerdir, toplevelname, dryrun=dryrun)
        else: 
             self._folder_extract(containerdir, toplevelname, dryrun=dryrun)



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

    tgz = "/var/dbbackup/mysqlhotcopy/belle7.nuu.edu.tw/tmp_offline_db/20130520_1353.tar.gz"
    t = Tar(tgz, toplevelname="tmp_offline_db")
    t.examine()
    #t.extract("/tmp/out")
    #t.examine()

    log.info(seconds)




