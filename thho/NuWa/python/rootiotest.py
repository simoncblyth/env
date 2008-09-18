#!/usr/bin/env python

import PyCintex

input = False
output = not input
use_aes = True

import GaudiPython as gp

app = gp.AppMgr(outputlevel=1)

# Use TES + AES instead of just AES
if use_aes:
    app.SvcOptMapping = [ 'DybDataSvc/EventDataSvc',
                          'EvtDataSvc/EventDataArchiveSvc',
                          ]

app.OutputLevel = 1
app.EvtMax = 3
if input:
    app.EvtSel =""
else:
    app.EvtSel = "NONE"         # yeah, I know, it's weird.

if input:
    app.ExtSvc += [ "RootIOEvtSelector/EventSelector" ]
    
app.ExtSvc += [ "RootIOCnvSvc" ]

per = app.service("EventPersistencySvc")
per.CnvServices = [ "RootIOCnvSvc" ];

eds = app.service("EventDataService")
eds.OutputLevel = 1

rio = app.property("RootIOCnvSvc")
rio.OutputLevel = 1
if input:
    rio.InputStreams = { "/Event/foo": "foo.root",
                         "default": "default.root" }
if output:
    rio.OutputStreams = { "/Event/foo": "foo.root",
                          "default": "default.root" }


if input:
    rioes = app.service("RootIOEvtSelector")
    rioes.OutputLevel = 1
else:
    spew = app.algorithm("spew")
    spew.OutputLevel = 1

app.TopAlg = [ ]

if not input:
    app.TopAlg += [ 'ObjectSpew/spew' ]

app.TopAlg += [ 'SimpleInputModule/sim' ]

if output:
    app.TopAlg += [ 'SimpleOuputModule/som' ]


sim = app.algorithm("sim")
sim.OutputLevel = 1

if output:
    som = app.algorithm("som")
    som.OutputLevel = 1

# Run...
app.initialize()
app.run(app.EvtMax)
