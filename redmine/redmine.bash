# === func-gen- : redmine/redmine fgp redmine/redmine.bash fgn redmine fgh redmine
redmine-src(){      echo redmine/redmine.bash ; }
redmine-source(){   echo ${BASH_SOURCE:-$(env-home)/$(redmine-src)} ; }
redmine-vi(){       vi $(redmine-source) ; }
redmine-env(){      elocal- ; }
redmine-usage(){
  cat << EOU
     redmine-src : $(redmine-src)
     redmine-dir : $(redmine-dir)



   http://bitnami.org/stack/redmine


 == Redmine Features ==

   REST API ?

   http://www.redmine.org/wiki/1/Rest_api

   REST API for Adding to timeseries is planned for 1.1 

   http://www.redmine.org/wiki/redmine/Rest_api_with_python


 == TODO ==

  Check out redmine in other places that github

       http://code.google.com/query/#q=redmine


 == plugins ==

    http://www.redmine.org/plugins
    https://github.com/search?type=Repositories&language=&q=redmine&repo=&langOverride=&x=0&y=0&start_value=1


  Redmine principals repos ...
    https://github.com/edavis10


  Add a tab 
    https://github.com/jamtur01/redmine_tab
    https://github.com/nbolton/redmine_wiki_tabs

  Opensearch is builtin with Trac
    https://github.com/meineerde/redmine_opensearch

  Commandline control 
    https://github.com/textgoeshere/redcmd

  For plugin dev
    https://github.com/edavis10/empty-redmine-plugin
    https://github.com/edavis10/redmine_plugin_support

  Tag issues/wiki-pages
    https://github.com/edavis10/redmine_tags
    https://github.com/friflaj/redmine_tagging
    https://github.com/ixti/redmine_tags

  Django inspectdb on a redmine DB + Django integration?
    https://github.com/lincolnloop/django-redmine
    https://github.com/codekoala/ponymine 
    https://github.com/zepheira/django-redmine


  LDAP
    https://github.com/edavis10/redmine_extra_ldap
    https://github.com/redivy/redmine_ldap_wiki_tags

  Fine grained access control 
    https://github.com/transitdk/Redmine-Repository-Control
    https://github.com/freedayko/redmine_repository_control

  ResT
    https://github.com/alphabetum/redmine_restructuredtext_formatter

  SQL reports  
    https://github.com/alantrick/redmine_sql_reports




EOU
}
redmine-dir(){ echo $(local-base)/env/redmine/redmine-redmine ; }
redmine-cd(){  cd $(redmine-dir); }
redmine-mate(){ mate $(redmine-dir) ; }
redmine-get(){
   local dir=$(dirname $(redmine-dir)) &&  mkdir -p $dir && cd $dir

}
