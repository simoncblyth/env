# === func-gen- : django/djsa workflow fgp django/djsa.bash fgn djsa fgh django
djsa-src(){      echo dj/djsa.bash ; }
djsa-source(){   echo ${BASH_SOURCE:-$(env-home)/$(djsa-src)} ; }
djsa-vi(){       vi $(djsa-source) ; }
djsa-env(){      elocal- ; }
djsa-usage(){
  cat << EOU
     djsa-src : $(djsa-src)
     djsa-dir : $(djsa-dir)







     djsa-tests 
           Ran 104 tests in 24.268s
           FAILED (failures=2) 
  

EOU
}
djsa-dir(){ echo $(local-base)/env/django/django-sqlalchemy/mainline ; }
djsa-cd(){  cd $(djsa-dir); }
djsa-mate(){ mate $(djsa-dir) ; }
djsa-get(){
   local dir=$(dirname $(djsa-dir)) &&  mkdir -p $dir && cd $dir
   git clone git://gitorious.org/django-sqlalchemy/mainline.git
}

djsa-ln(){ 
    python- 
    python-ln $(djsa-dir)/django_sqlalchemy 
}

djsa-noseplugin(){
   djsa-cd 
   cd django_sqlalchemy/test/nose-django-sqlalchemy
   sudo python setup.py install
   nosetests -p
}

djsa-blogtests(){
  dj-
  DJANGO_PROJECT=djsa dj-
  cd $(dj-dir-)/djsa
  python blog/tests.py

  echo "select * from blog_post ; " | dj-mysql
}


djsa-tests(){
   local msg="=== $FUNCNAME :"
   echo $msg arghh the plugin is changing DJANGO_SETTINGS_MODULE to "settings" ... so must run tests from the tests dir
   djsa-cd ; cd tests
   nosetests -v -s --with-django-sqlalchemy --with-doctest --doctest-extension=test 
}


djsa-manual-fix(){

   echo $msg need to do this for tests to succeed with a recent django 
   cat << EOF

[blyth@belle7 backend]$ git diff
diff --git a/django_sqlalchemy/backend/base.py b/django_sqlalchemy/backend/base.py
index 42a1adf..b5c85f2 100644
--- a/django_sqlalchemy/backend/base.py
+++ b/django_sqlalchemy/backend/base.py
@@ -33,7 +33,7 @@ class SQLAlchemyDatabaseWrapper(BaseDatabaseWrapper):
         self.introspection = DatabaseIntrospection(self)
         self.validation = BaseDatabaseValidation()
 
-    def _cursor(self, settings):
+    def _cursor(self):
         return session.connection().connection.cursor()
 

EOF

}
