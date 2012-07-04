"""
https://github.com/fabric/fabric/issues/611
"""

import fabric.network

# use ssh configuration even if env.user and env.port habe been modified
def normalize(host_string, omit_port=False):
    from fabric.state import env
    if not host_string:
        return ('', '') if omit_port else ('', '', '')
    r = fabric.network.parse_host_string(host_string)
    host = r['host']
    user = env.user or env.local_user
    port = env.port or env.default_port
    conf = fabric.network.ssh_config(host_string)
    if 'user' in conf:
        user = conf['user']
    if 'port' in conf:
        port = conf['port']
    if 'hostname' in conf:
        host = conf['hostname']
    user = r['user'] or user
    port = r['port'] or port
    if omit_port:
        return user, host
    return user, host, port

fabric.network.normalize = normalize
