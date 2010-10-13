
from django.db import connection
from django.conf import settings
from django.contrib.auth.models import User

from django.contrib.auth.backends import ModelBackend


from env.trac.auth import BasicAuthentication 

def basic_check_password( auth_user_file , username , password ):
    """
       Doing it all every time for simplicity and immediate adaption to changed auth_user_file
    """
    ba = BasicAuthentication( auth_user_file )
    return ba.test( username , password )
        


## MySQL 41 char password() correspond to this  
from hashlib import sha1
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
    assert settings.DATABASE_ENGINE == 'mysql'
    cursor = connection.cursor()
    if cursor.execute("select password(%s)", [ password ] ) == 1:
        return cursor.fetchone()[0]
    return "encrypted_password_failed" 
        
class AuthUserFileBackend(ModelBackend):
    """
        This specialises the default ModelBackend, but only steps in 
        if authentication with the base Backend (standard django Users) 
        fails ... in which case authentication against the AUTH_USER_FILE is
        attempted. In the case of valid credentials new django users are created
        but with a password that django will not recognise...   

        Usage :

           1) in settings point to this backend with : 
                    AUTHENTICATION_BACKENDS = ( 'env.trac.dj.backends.AuthUserFileBackend',)

               which overrides the default, which may not be present in your settings of:
                    AUTHENTICATION_BACKENDS = ( 'django.contrib.auth.backends.ModelBackend',)

           2)  also in settings 
                     AUTH_USER_FILE=/absolute/path/to/auth_user_file



          Based on example from 
               http://docs.djangoproject.com/en/dev/topics/auth/#writing-an-authentication-backend

          Note than on creating a new user we can set the password
          to anything, because it won't be checked; the password
          from settings.AUTH_USER_FILE will.
       
    """ 
    def authenticate(self, username=None, password=None):
        
        user = ModelBackend.authenticate( self, username, password )
        if user:
           return user

        valid = basic_check_password( settings.AUTH_USER_FILE,  username, password )
        if valid:
            try:
                user = User.objects.get(username=username)
            except User.DoesNotExist:
                user = User(username=username, password=mysql_password(password))
                user.is_staff = True   ## MUST TO staff TO ENABLE ACCESS TO ADMIN SITE 
                user.is_superuser = False
                user.save()
            return user
        return None

