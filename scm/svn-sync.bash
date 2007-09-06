
svn-sync-init(){

    local cmd="svnsync initialize $SCM_URL/repos/dybsvn $DYBSVN --username $SVN_SYNC_USER --password $SVN_SYNC_PASS"
    echo $cmd
    eval $cmd

}


svn-sync-synchronize(){

    local cmd="svnsync synchronize $SCM_URL/repos/dybsvn --username $SVN_SYNC_USER --password $SVN_SYNC_PASS"
    echo $cmd
    eval $cmd
                     
}