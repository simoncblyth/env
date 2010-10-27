
from django.db import connection
from django.conf import settings
from django.contrib.auth.models import User
#from django.contrib.auth.backends import ModelBackend

from env.trac.auth import BasicAuthentication 


import logging as log
log.basicConfig(
    level = log.INFO,
    format = '%(asctime)s %(levelname)s %(message)s',
)


def basic_check_password( auth_user_file , username , password ):
    """
       Doing it all every time for simplicity and immediate adaption to changed auth_user_file
    """
    ba = BasicAuthentication( auth_user_file )
    return ba.test( username , password )
        


## MySQL 41 char password() correspond to this  
try:
    from hashlib import sha1
except ImportError:
    from sha import sha as sha1
mysql_password41_ = lambda pw:"*"+sha1(sha1(pw).digest()).hexdigest().upper()


def mysql_password(password):
    """
        Setting the django User password to the mysql encrypted password
        even though django will not be looking at this for authentication, can 
        make use of this for generation of mysql user grant scripts ... allowing 
        to propagate to mysql users

          FOR BETTER SECURITY CHECK YOU DO NOT HAVE old_passwords=1 in /etc/my.cnf 
              * at expense of loosing compatibility pre 4.1
              * http://dev.mysql.com/doc/refman/5.1/en/password-hashing.html
       
               echo "select password('hello')" | mysql 
                   password('hello')
                   70de51425df9d787
        
               echo "select password('hello')" | mysql 
                   password('hello')
                   *6B4F89A54E2D27ECD7E8DA05B4AB8FD9D1D8B119
      

    """
    #assert settings.DATABASE_ENGINE == 'mysql'
    cursor = connection.cursor()
    if cursor.execute("select password(%s)", [ password ] ) == 1:
        return cursor.fetchone()[0]
    return "encrypted_password_failed" 
        
class AuthUserFileBackend:
    """
        An authentication Backend authenticating against 
        SVN/Trac username/passwords in settings.AUTH_USER_FILE
        Only invoked if authentication with prior backends 
        in the sequence : settings.AUTHENTICATION_BACKENDS all fail

        In the case of valid credentials new django users are created
        but with a password that django will not recognise thus future logins
        will fail other authentications and pass thru to this one
        (assuming configured after the default backend in settings) 

        CAUTION : 
             it is best not to have username overlap between native 
             django users and AUTH_USER_FILE ones to avoid confusion/changed passwords

        Usage :

           1) in settings point to this backend with : 
                    AUTHENTICATION_BACKENDS = ( 
                           'django.contrib.auth.backends.ModelBackend',  ## the default 
                           'env.trac.dj.backends.AuthUserFileBackend',
                         )

           2)  also in settings 
                     AUTH_USER_FILE=/absolute/path/to/auth_user_file



          Based on example from 
               http://docs.djangoproject.com/en/dev/topics/auth/#writing-an-authentication-backend

          Note than on creating a new user we can set the password
          to anything, because it won't be checked; the password
          from settings.AUTH_USER_FILE will.
       
    """
    supports_object_permissions = False
    supports_anonymous_user = False

    def authenticate(self, username=None, password=None):
        """
           Observing ...
                mysql> select * from auth_user ;
           shows that this usurps pre-existing django native users by changing their passwords

           After changing passwords to the AUTH_USER_FILE one, django will fail to authenticate
           natively and hence will be passed along here

        """
        valid = basic_check_password( settings.AUTH_USER_FILE,  username, password )
        log.info( "authentication of  %s from %s valid %s " , username, self.__class__.__name__ , valid  ) 
        if valid:
            pw = mysql_password(password)
            try:
                user = User.objects.get(username=username)
                log.info( "authentication, preexisting User %s  " , user  ) 
                if user.password != pw:         ## update django hashed mysql password on changes 
                    user.password = pw
                    log.info( "authentication, updating password of preexisting User %s  " , user  ) 
                    user.save()
            except User.DoesNotExist:
                user = User(username=username, password=pw )
                log.info( "authentication, creating new User %s  " , user  ) 
                user.is_staff = True             ## needs to be staff TO ENABLE ACCESS TO ADMIN SITE 
                user.is_superuser = False
                user.save()
            return user
        log.info("authentication failed for %s ", username )
        return None

    def get_user(self, user_id):
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None





