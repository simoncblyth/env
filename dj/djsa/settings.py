# Django settings for djsa project.

DEBUG = True
TEMPLATE_DEBUG = DEBUG

ADMINS = (
    # ('Your Name', 'your_email@domain.com'),
)

MANAGERS = ADMINS



from env.base.private import Private
p = Private()



sa = True 
if sa:
    DATABASE_ENGINE = 'django_sqlalchemy.backend' 
    DJANGO_SQLALCHEMY_DBURI = p('DATABASE_URL')
    DJANGO_SQLALCHEMY_ECHO = False 
else:
    DATABASE_ENGINE = p('DATABASE_ENGINE')           # 'postgresql_psycopg2', 'postgresql', 'mysql', 'sqlite3' or 'oracle'.
    DATABASE_NAME   = p('DATABASE_NAME')             # Or path to database file if using sqlite3.
    DATABASE_USER   = p('DATABASE_USER')             # Not used with sqlite3.
    DATABASE_PASSWORD = p('DATABASE_PASSWORD')         # Not used with sqlite3.
    DATABASE_HOST = p('DATABASE_HOST')             # Set to empty string for localhost. Not used with sqlite3.
    DATABASE_PORT = p('DATABASE_PORT')             # Set to empty string for default. Not used with sqlite3.

# Local time zone for this installation. Choices can be found here:
# http://en.wikipedia.org/wiki/List_of_tz_zones_by_name
# although not all choices may be available on all operating systems.
# If running in a Windows environment this must be set to the same as your
# system time zone.
TIME_ZONE = 'Asia/Taipei'

# Language code for this installation. All choices can be found here:
# http://www.i18nguy.com/unicode/language-identifiers.html
LANGUAGE_CODE = 'en-us'

SITE_ID = 1

# If you set this to False, Django will make some optimizations so as not
# to load the internationalization machinery.
USE_I18N = True

# Absolute path to the directory that holds media.
# Example: "/home/media/media.lawrence.com/"
MEDIA_ROOT = ''

# URL that handles the media served from MEDIA_ROOT. Make sure to use a
# trailing slash if there is a path component (optional in other cases).
# Examples: "http://media.lawrence.com", "http://example.com/media/"
MEDIA_URL = ''

# URL prefix for admin media -- CSS, JavaScript and images. Make sure to use a
# trailing slash.
# Examples: "http://foo.com/media/", "/media/".
ADMIN_MEDIA_PREFIX = '/media/'

# Make this unique, and don't share it with anybody.
SECRET_KEY = 'qw(+1^oulf7jf72e*4u91#wrkeqy8v0ahiohrf+u7s7bb6_7jc'

# List of callables that know how to import templates from various sources.
TEMPLATE_LOADERS = (
    'django.template.loaders.filesystem.load_template_source',
    'django.template.loaders.app_directories.load_template_source',
#     'django.template.loaders.eggs.load_template_source',
)

MIDDLEWARE_CLASSES = (
    'django.middleware.common.CommonMiddleware',
)

## required for the admin 
MIDDLEWARE_CLASSES += (
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
)



ROOT_URLCONF = 'djsa.urls'

TEMPLATE_DIRS = (
    # Put strings here, like "/home/html/django_templates" or "C:/www/django/templates".
    # Always use forward slashes, even on Windows.
    # Don't forget to use absolute paths, not relative paths.
)


INSTALLED_APPS = ()

## enable the admin ...   auth,contenttypes and sessions are required by the admin 
admin = False
if admin:INSTALLED_APPS += (
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.sites',
    'django.contrib.admin',
)

if DATABASE_ENGINE.startswith('django_sqlalchemy'):
    INSTALLED_APPS += ('django_sqlalchemy',)

INSTALLED_APPS += ('blog',)
INSTALLED_APPS += ('django_extensions',)



