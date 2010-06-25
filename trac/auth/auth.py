# -*- coding: utf-8 -*-
#
# Copyright (C) 2003-2008 Edgewall Software
# Copyright (C) 2003-2005 Jonas Borgström <jonas@edgewall.com>
# All rights reserved.
#
# This software is licensed as described in the file COPYING, which
# you should have received as part of this distribution. The terms
# are also available at http://trac.edgewall.org/wiki/TracLicense.
#
# This software consists of voluntary contributions made by many
# individuals. For the exact contribution history, see the revision
# history and logs, available at http://trac.edgewall.org/log/.
#
# Author: Jonas Borgström <jonas@edgewall.com>

try:
    from base64 import b64decode
except ImportError:
    from base64 import decodestring as b64decode
try:
    import threading
except ImportError:
    import dummy_threading as threading
import md5
import os
import re
import sys
import time
import urllib2

from util import hex_entropy, md5crypt

class HTTPAuthentication(object):

    def do_auth(self, environ, start_response):
        raise NotImplementedError


class PasswordFileAuthentication(HTTPAuthentication):
    def __init__(self, filename):
        self.filename = filename
        self.mtime = os.stat(filename).st_mtime
        self.load(self.filename)
        self._lock = threading.Lock()

    def check_reload(self):
        self._lock.acquire()
        try:
            mtime = os.stat(self.filename).st_mtime
            if mtime > self.mtime:
                self.mtime = mtime
                self.load(self.filename)
        finally:
            self._lock.release()

class BasicAuthentication(PasswordFileAuthentication):

    def __init__(self, htpasswd, realm="realm"):
        self.realm = realm
        try:
            import crypt
            self.crypt = crypt.crypt
        except ImportError:
            try:
                import fcrypt
                self.crypt = fcrypt.crypt
            except ImportError:
                self.crypt = None
        PasswordFileAuthentication.__init__(self, htpasswd)

    def load(self, filename):
        self.hash = {}
        fd = open(filename, 'r')
        for line in fd:
            line = line.strip()
            if not line:
                continue
            try:
                u, h = line.split(':')
            except ValueError:
                print >>sys.stderr, 'Warning: invalid password line in %s: %s' \
                                    % (filename, line)
                continue
            if '$' in h or self.crypt:
                self.hash[u] = h
            else:
                print >>sys.stderr, 'Warning: cannot parse password for ' \
                                    'user "%s" without the "crypt" module' % u

        if self.hash == {}:
            print >> sys.stderr, "Warning: found no users in file:", filename

    def test(self, user, password):
        self.check_reload()
        the_hash = self.hash.get(user)
        if the_hash is None:
            return False

        if not '$' in the_hash:
            return self.crypt(password, the_hash[:2]) == the_hash

        magic, salt = the_hash[1:].split('$')[:2]
        magic = '$' + magic + '$'
        return md5crypt(password, salt, magic) == the_hash

    def do_auth(self, environ, start_response):
        header = environ.get('HTTP_AUTHORIZATION')
        if header and header.startswith('Basic'):
            auth = b64decode(header[6:]).split(':')
            if len(auth) == 2:
                user, password = auth
                if self.test(user, password):
                    return user

        start_response('401 Unauthorized',
                       [('WWW-Authenticate', 'Basic realm="%s"'
                         % self.realm)])('')


class DigestAuthentication(PasswordFileAuthentication):
    """A simple HTTP digest authentication implementation (RFC 2617)."""

    MAX_NONCES = 100

    def __init__(self, htdigest, realm):
        self.active_nonces = []
        self.realm = realm
        PasswordFileAuthentication.__init__(self, htdigest)

    def load(self, filename):
        """Load account information from apache style htdigest files, only
        users from the specified realm are used
        """
        self.hash = {}
        fd = open(filename, 'r')
        for line in fd.readlines():
            line = line.strip()
            if not line:
                continue
            try:
                u, r, a1 = line.split(':')
            except ValueError:
                print >>sys.stderr, 'Warning: invalid digest line in %s: %s' \
                                    % (filename, line)
                continue
            if r == self.realm:
                self.hash[u] = a1
        if self.hash == {}:
            print >> sys.stderr, "Warning: found no users in realm:", self.realm
        
    def parse_auth_header(self, authorization):
        values = {}
        for value in urllib2.parse_http_list(authorization):
            n, v = value.split('=', 1)
            if v[0] == '"' and v[-1] == '"':
                values[n] = v[1:-1]
            else:
                values[n] = v
        return values

    def send_auth_request(self, environ, start_response, stale='false'):
        """Send a digest challange to the browser. Record used nonces
        to avoid replay attacks.
        """
        nonce = hex_entropy()
        self.active_nonces.append(nonce)
        if len(self.active_nonces) > self.MAX_NONCES:
            self.active_nonces = self.active_nonces[-self.MAX_NONCES:]
        start_response('401 Unauthorized',
                       [('WWW-Authenticate',
                        'Digest realm="%s", nonce="%s", qop="auth", stale="%s"'
                        % (self.realm, nonce, stale))])('')

    def do_auth(self, environ, start_response):
        header = environ.get('HTTP_AUTHORIZATION')
        if not header or not header.startswith('Digest'):
            self.send_auth_request(environ, start_response)
            return None

        auth = self.parse_auth_header(header[7:])
        required_keys = ['username', 'realm', 'nonce', 'uri', 'response',
                         'nc', 'cnonce']
        # Invalid response?
        for key in required_keys:
            if not auth.has_key(key):
                self.send_auth_request(environ, start_response)
                return None
        # Unknown user?
        self.check_reload()
        if not self.hash.has_key(auth['username']):
            self.send_auth_request(environ, start_response)
            return None

        kd = lambda x: md5.md5(':'.join(x)).hexdigest()
        a1 = self.hash[auth['username']]
        a2 = kd([environ['REQUEST_METHOD'], auth['uri']])
        # Is the response correct?
        correct = kd([a1, auth['nonce'], auth['nc'],
                      auth['cnonce'], auth['qop'], a2])
        if auth['response'] != correct:
            self.send_auth_request(environ, start_response)
            return None
        # Is the nonce active, if not ask the client to use a new one
        if not auth['nonce'] in self.active_nonces:
            self.send_auth_request(environ, start_response, stale='true')
            return None
        self.active_nonces.remove(auth['nonce'])
        return auth['username']
