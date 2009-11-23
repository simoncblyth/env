# notifymq-ipython mq_handlebytes.py
import ROOT
ROOT.gSystem.Load("lib/libnotifymq.so")
ROOT.MQ.test_handlebytes()


