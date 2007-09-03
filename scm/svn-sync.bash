
svn-sync-init(){

    svnsync initialize $SCM_URL/repos/dybsvn $DYBSVN --username $SVN_SYNC_USER --password $SVN_SYNC_PASS

}


svn-sync-synchronize(){

    svnsync synchronize $SCM_URL/repos/dybsvn --username $SVN_SYNC_USER --password $SVN_SYNC_PASS
                     
}