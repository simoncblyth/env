hg convert
============


::

    delta:env blyth$ find . -name '*.pyc' -exec rm -f {} \; 
    delta:env blyth$ diff -r --brief env /tmp/ee/env
    diff: env: No such file or directory
    delta:env blyth$ diff -r --brief ~/env /tmp/ee/env
    Only in /tmp/ee/env: .hg
    Only in /Users/blyth/env: .svn
    Only in /Users/blyth/env/AbtViz: tests
    Only in /Users/blyth/env: _build
    Only in /Users/blyth/env: _docs
    Only in /Users/blyth/env: _static
    Only in /Users/blyth/env: beizhen
    Only in /Users/blyth/env/bin: dna.py
    Only in /Users/blyth/env/bin: issues_json.py
    Only in /Users/blyth/env/bin: realpath
    Only in /Users/blyth/env: bzhu
    Only in /Users/blyth/env/cuda: .cuda_context_cleanup.rst.swp
    Only in /Users/blyth/env/dj/dybsite/dbi: fixtures
    Files /Users/blyth/env/doc/sphinx.bash and /tmp/ee/env/doc/sphinx.bash differ
    Files /Users/blyth/env/env.bash and /tmp/ee/env/env.bash differ
    Only in /Users/blyth/env/geant4: g4beta
    Only in /Users/blyth/env: gpu
    diff: /Users/blyth/env/graphics/collada/colladadom/testColladaDOM/run: No such file or directory
    diff: /tmp/ee/env/graphics/collada/colladadom/testColladaDOM/run: No such file or directory
    Only in /Users/blyth/env/graphics/transformations: .transformations.bash.swp
    Only in /Users/blyth/env/graphics/webgl: webglbook
    Only in /Users/blyth/env/graphics: x3d
    Only in /Users/blyth/env/hg: .hg.bash.swp
    Files /Users/blyth/env/hg/hg.bash and /tmp/ee/env/hg/hg.bash differ
    Only in /Users/blyth/env/hub: _static
    Only in /Users/blyth/env/hub: _templates
    Only in /Users/blyth/env: legacy
    Only in /Users/blyth/env: liteng
    Only in /Users/blyth/env: litsh08
    Only in /Users/blyth/env/macros: aberdeen
    Only in /Users/blyth/env: pip
    Only in /Users/blyth/env/root/tutorials: net
    Only in /Users/blyth/env: seed
    Only in /Users/blyth/env: setup
    Only in /Users/blyth/env/svn: bindings
    Only in /Users/blyth/env: svn_
    Only in /Users/blyth/env/thho/NuWa/AcrylicOpticalSim: src
    Only in /Users/blyth/env/trac/dj: tests
    Only in /Users/blyth/env/trac/migration: .overview.rst.swp
    Only in /tmp/ee/env/trac/migration: check_issues_json.py
    Only in /Users/blyth/env/trac/migration: issues_json.py
    Files /Users/blyth/env/trac/migration/overview.rst and /tmp/ee/env/trac/migration/overview.rst differ
    Files /Users/blyth/env/trac/migration/trac2bitbucket.bash and /tmp/ee/env/trac/migration/trac2bitbucket.bash differ
    Only in /Users/blyth/env/trac/migration: tracmigrate.bash
    Files /Users/blyth/env/trac/migration/tracwikidump.py and /tmp/ee/env/trac/migration/tracwikidump.py differ
    delta:env blyth$ 

