Context Reporting Plugin
=========================

Currently testing on N in a builtin fashion.::

    plugins=/data1/env/local/dyb/external/nose/0.11.4_python2.7/i686-slc5-gcc41-dbg/lib/python2.7/site-packages/nose/plugins
    cp ctx.py $plugins/
    vi $plugins/builtin.py

Enable the builtin via the builtins list::

    ('nose.plugins.collect', 'CollectOnly'),
    #('nose.plugins.ctx', 'Ctx'),
    )


enabling plugin
-----------------

Can be enabled in :file:`~/.noserc` but getting users to do something like that 
is non-trivial

nosetests is a very simple script::


    [blyth@belle7 ~]$ cat  /data1/env/local/dyb/external/nose/0.11.4_python2.7/i686-slc5-gcc41-dbg/bin/nosetests
    #!/data1/env/local/dyb/external/Python/2.7/i686-slc5-gcc41-dbg/bin/python

    from nose import main

    if __name__ == '__main__':
        main()


Registering a plugin without setuptools
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Setuptools is always painful, programatic registration 
via custom nose runner looks attractive:

.. code-block:: python
    
    import nose
    from yourplugin import YourPlugin
    
    if __name__ == '__main__':
        nose.main(addplugins=[YourPlugin()])


