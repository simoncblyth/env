# Django settings for dybprj project.

from private import Private 
p = Private()

from env.offline.dbconf import DBConf

import os
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DJYLT_DIR = p('DJYLT_DIR')
DEFAULT_CONTENT_TYPE =  "application/xhtml+xml"   # default mimetype, needed for inline SVG to show up 

DEBUG = True
TEMPLATE_DEBUG = DEBUG

OLIVE_SERVER_HOST = p('OLIVE_SERVER_HOST')
OLIVE_SERVER_PORT = p('OLIVE_SERVER_PORT')
OLIVE_AMQP_EXCHANGE = p('OLIVE_AMQP_EXCHANGE') 
OLIVE_KEY_FUNC = lambda k:'olive.%s.string' % k


ADMINS = (
    ( p('ADMIN_NAME') ,  p('ADMIN_EMAIL') ),
)

MANAGERS = ADMINS

DATABASES = dict([ (k,DBConf(k,verbose=False).django) for k in "default client".split() ] )  



# Local time zone for this installation. Choices can be found here:
# http://en.wikipedia.org/wiki/List_of_tz_zones_by_name
# although not all choices may be available on all operating systems.
# On Unix systems, a value of None will cause Django to use the same
# timezone as the operating system.
# If running in a Windows environment this must be set to the same as your
# system time zone.
TIME_ZONE = 'Asia/Taipei'

# Language code for this installation. All choices can be found here:
# http://www.i18nguy.com/unicode/language-identifiers.html
LANGUAGE_CODE = 'en-us'

SITE_ID = 1
SITE_DOMAIN = p('SITE_DOMAIN')

# If you set this to False, Django will make some optimizations so as not
# to load the internationalization machinery.
USE_I18N = True

# If you set this to False, Django will not format dates, numbers and
# calendars according to the current locale
USE_L10N = True

# Absolute path to the directory that holds media.
# Example: "/home/media/media.lawrence.com/"
MEDIA_ROOT = p('MEDIA_ROOT')

# URL that handles the media served from MEDIA_ROOT. Make sure to use a
# trailing slash if there is a path component (optional in other cases).
# Examples: "http://media.lawrence.com", "http://example.com/media/"
MEDIA_URL = p('MEDIA_URL')

# URL prefix for admin media -- CSS, JavaScript and images. Make sure to use a
# trailing slash.
# Examples: "http://foo.com/media/", "/media/".
ADMIN_MEDIA_PREFIX = '/media/'

# Make this unique, and don't share it with anybody.
SECRET_KEY = p('ADMIN_SECRET_KEY')

# List of callables that know how to import templates from various sources.
TEMPLATE_LOADERS = (
    'django.template.loaders.filesystem.Loader',
    'django.template.loaders.app_directories.Loader',
#     'django.template.loaders.eggs.Loader',
)

MIDDLEWARE_CLASSES = (
    'django.middleware.common.CommonMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
)

ROOT_URLCONF = 'dybprj.urls'

TEMPLATE_DIRS = (
    PROJECT_DIR + "/templates" , 
    DJYLT_DIR + "/templates" ,
)

INSTALLED_APPS = (
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.sites',
    'django.contrib.messages',
    'django.contrib.admin',
    'django.contrib.admindocs',
    'django.contrib.comments',
    'guardian',                   ## see guardian- 
    'dybprj.db',                  ## see dybprj-
    'dybprj.olive',
    'dybprj.job',
    'dybprj.cust_comments',
    'django_extensions',          ## see djext-
)


COMMENTS_APP = 'dybprj.cust_comments'   ## make email/url/name fields optional 



AUTHENTICATION_BACKENDS = (
    'django.contrib.auth.backends.ModelBackend',   # the default, authenticating against native Users 
    'guardian.backends.ObjectPermissionBackend',   # this always returns None to authenticate ... so will pass on below 
    'env.trac.dj.backends.AuthUserFileBackend',    # authenticate against AUTH_USER_FILE credentials
)

AUTH_USER_FILE=p('AUTH_USER_FILE')    # used by AuthUserFileBackend

ANONYMOUS_USER_ID = -1     # used by guardian



# A sample logging configuration. The only tangible logging
# performed by this configuration is to send an email to
# the site admins on every HTTP 500 error.
# See http://docs.djangoproject.com/en/dev/topics/logging for
# more details on how to customize your logging configuration.
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'mail_admins': {
            'level': 'ERROR',
            'class': 'django.utils.log.AdminEmailHandler'
        }
    },
    'loggers': {
        'django.request':{
            'handlers': ['mail_admins'],
            'level': 'ERROR',
            'propagate': True,
        },
    }
}
