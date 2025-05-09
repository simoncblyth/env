#!/usr/bin/env python

#
#   http://www.chiark.greenend.org.uk/~sgtatham/svn.html
#   http://www.tartarus.org/~simon-anonsvn/viewcvs.cgi/svn-tools/svndumpfilter2?view=markup
#
#
# Utility to filter a dump file of a Subversion repository to
# produce a dump file describing only specified subdirectories of
# the tree contained in the original one. This is similar in
# concept to the official tool `svndumpfilter', but it's able to
# cope with revisions which copy files into the area of interest
# from outside it (in which situation a Node-copyfrom won't be
# valid in the output dump file). However, in order to support
# this, svndumpfilter2 requires access via `svnlook' to the
# original repository from which the input dump file was produced.
#
# Usage:
#
#     svndumpfilter source-repository regexp [regexp...]
#
# This command expects to receive a Subversion dump file on
# standard input, which must correspond to the Subversion
# repository pointed to by the first argument. It outputs a
# filtered dump file on standard output.
#
# `source-repository': The first argument must be a pathname to a
# _local_ Subversion repository. That is, it isn't a Subversion URL
# (beginning with http:// or svn:// or anything else like that);
# it's a simple local pathname (absolute or relative). A simple
# test to see if it's a valid pathname is to pass it as an argument
# to `svnlook tree'. If that succeeds, it's also a valid first
# argument to svndumpfilter2.
#
# `regexp': The remaining arguments are used to select directory
# names from the top level of the repository's internal directory
# tree. Any directory matching any of the regexps will be
# considered `interesting' and copied into the output dump file;
# any directory not matching will not. Matching is performed at the
# top level only: it is not currently possible to selectively
# include a subset of second-level directories with a common
# parent.
#
# For example, this command...
#
#     svndumpfilter2 /home/svnadmin/myrepos foo bar baz quu+x
#
# ... will read a dump file on standard input, and output one on
# standard output which contains only the subdirectories `foo',
# `bar', `baz', `quux', `quuux', `quuuux', etc.
#
# You will probably usually want to use svndumpfilter2 in
# conjunction with the production of the dump file in the first
# place, like this:
#
#     svnadmin dump /home/svnadmin/myrepos | \
#         svndumpfilter2 /home/svnadmin/myrepos foo bar baz quu+x > msv.dump

import sys
import os
import re
import string
import types
import md5

# Quoting function which should render any string impervious to
# POSIX shell metacharacter expansion.
def quote(word):
    return "'" + string.replace(word, "'", "'\\''") + "'"

# First, the sensible way to deal with a pathname is to split it
# into pieces at the slashes and thereafter treat it as a list.
def splitpath(s):
    list = string.split(s, "/")
    # Simplest way to remove all empty elements!
    try:
	while 1:
	    list.remove("")
    except ValueError:
	pass
    return list

def joinpath(list, prefix=""):
    return prefix + string.join(list, "/")

def catpath(path1, path2, prefix=""):
    return joinpath(splitpath(path1) + splitpath(path2), prefix)

# Decide whether a pathname is interesting or not.
class InterestingPaths:
    def __init__(self, args):
	self.res = []
	for a in args:
	    self.res.append(re.compile(a))
    def interesting(self, path):
	a = splitpath(path)
	if len(a) == 0:
	    # It's possible that the path may have no elements at
	    # all, in which case we can't match on its first
	    # element. This generally occurs when svn properties
	    # are being changed on the root of the repository; we
	    # consider those to be always interesting and never
	    # filter them out.
	    return 1
	for r in self.res:
	    if r.match(a[0]):
		return 1
	return 0

# A class and some functions to handle a single lump of
# RFC822-ish-headers-plus-data read from an SVN dump file.

class Lump:
    def __init__(self):
	self.hdrlist = []
	self.hdrdict = {}
	self.prop = ""
	self.text = ""
	self.extant = 1
	self.props = [[], {}]
    def sethdr(self, key, val):
	if not self.hdrdict.has_key(key):
	    self.hdrlist.append(key)
	self.hdrdict[key] = val
    def delhdr(self, key):
	if self.hdrdict.has_key(key):
	    del self.hdrdict[key]
	    self.hdrlist.remove(key)
    def propparse(self):
	index = 0
	while 1:
	    if self.prop[index:index+2] == "K ":
		wantval = 1
	    elif self.prop[index:index+2] == "D ":
		wantval = 0
	    elif self.prop[index:index+9] == "PROPS-END":
		break
	    else:
		raise "Unrecognised record in props section"
	    nlpos = string.find(self.prop, "\n", index)
	    assert nlpos > 0
	    namelen = string.atoi(self.prop[index+2:nlpos])
	    assert self.prop[nlpos+1+namelen] == "\n"
	    name = self.prop[nlpos+1:nlpos+1+namelen]
	    index = nlpos+2+namelen
	    if wantval:
		assert self.prop[index:index+2] == "V "
		nlpos = string.find(self.prop, "\n", index)
		assert nlpos > 0
		proplen = string.atoi(self.prop[index+2:nlpos])
		assert self.prop[nlpos+1+proplen] == "\n"
		prop = self.prop[nlpos+1:nlpos+1+proplen]
		index = nlpos+2+proplen
	    else:
		prop = None
	    self.props[0].append(name)
	    self.props[1][name] = prop
    def setprop(self, key, val):
	if not self.props[1].has_key(key):
	    self.props[0].append(key)
	self.props[1][key] = val
    def delprop(self, key):
	if self.props[1].has_key(key):
	    del self.props[1][key]
	    self.props[0].remove(key)
    def correct_headers(self):
	# First reconstitute the properties block.
	self.prop = ""
	if len(self.props[0]) > 0:
	    for key in self.props[0]:
		val = self.props[1][key]
		if val == None:
		    self.prop = self.prop + "D %d" % len(key) + "\n" + key + "\n"
		else:
		    self.prop = self.prop + "K %d" % len(key) + "\n" + key + "\n"
		    self.prop = self.prop + "V %d" % len(val) + "\n" + val + "\n"
	    self.prop = self.prop + "PROPS-END\n"
	# Now fix up the content length headers.
	if len(self.prop) > 0:
	    self.sethdr("Prop-content-length", str(len(self.prop)))
	else:
	    self.delhdr("Prop-content-length")
	# Only fiddle with the md5 if we're not doing a delta.
	if self.hdrdict.get("Text-delta", "false") != "true":
	    if len(self.text) > 0:
		self.sethdr("Text-content-length", str(len(self.text)))
		m = md5.new()
		m.update(self.text)
		self.sethdr("Text-content-md5", m.hexdigest())
	    else:
		self.delhdr("Text-content-length")
		self.delhdr("Text-content-md5")
	if len(self.prop) > 0 or len(self.text) > 0:
	    self.sethdr("Content-length", str(len(self.prop)+len(self.text)))
	else:
	    self.delhdr("Content-length")

def read_rfc822_headers(f):
    ret = Lump()
    while 1:
	s = f.readline()
	if s == "":
	    return None # end of file
	if s == "\n":
	    if len(ret.hdrlist) > 0:
		break # newline after headers ends them
	    else:
		continue # newline before headers is simply ignored
	if s[-1:] == "\n": s = s[:-1]
	colon = string.find(s, ":")
	assert colon > 0
	assert s[colon:colon+2] == ": "
	key = s[:colon]
	val = s[colon+2:]
	ret.sethdr(key, val)
    return ret

def read_lump(f):
    lump = read_rfc822_headers(f)
    if lump == None:
	return None
    pcl = string.atoi(lump.hdrdict.get("Prop-content-length", "0"))
    tcl = string.atoi(lump.hdrdict.get("Text-content-length", "0"))
    if pcl > 0:
	lump.prop = f.read(pcl)
	lump.propparse()
    if tcl > 0:
	lump.text = f.read(tcl)
    return lump

def write_lump(f, lump):
    if not lump.extant:
	return
    lump.correct_headers()
    for key in lump.hdrlist:
	val = lump.hdrdict[key]
	f.write(key + ": " + val + "\n")
    f.write("\n")
    f.write(lump.prop)
    f.write(lump.text)
    if lump.hdrdict.has_key("Prop-content-length") or \
    lump.hdrdict.has_key("Text-content-length") or \
    lump.hdrdict.has_key("Content-length"):
	f.write("\n")

# Higher-level class that makes use of the above to filter dump
# file fragments a whole revision at a time.

class Filter:
    def __init__(self, paths):
	self.revisions = {}
	self.paths = paths

    def tweak(self, revhdr, contents):
	contents2 = []
	for lump in contents:
	    action = lump.hdrdict["Node-action"]
	    path = lump.hdrdict["Node-path"]

	    if not self.paths.interesting(path):
		continue # boooring

	    need = 1 # we need to do something about this lump

	    if action == "add":
		if lump.hdrdict.has_key("Node-copyfrom-path"):
		    srcrev = string.atoi(lump.hdrdict["Node-copyfrom-rev"])
		    srcpath = lump.hdrdict["Node-copyfrom-path"]
		    if not self.paths.interesting(srcpath):
			# Copy from a boring path to an interesting
			# one, meaning we must use svnlook to
			# extract the subtree and convert it into
			# lumps.
			treecmd = "svnlook tree -r%d %s %s" % \
			(srcrev, quote(repos), quote(srcpath))
			tree = os.popen(treecmd, "r")
			pathcomponents = []
			while 1:
			    treeline = tree.readline()
			    if treeline == "": break
			    if treeline[-1:] == "\n": treeline = treeline[:-1]
			    subdir = 0
			    while treeline[-1:] == "/":
				subdir = 1
				treeline = treeline[:-1]
			    depth = 0
			    while treeline[:1] == " ":
				depth = depth + 1
				treeline = treeline[1:]
			    pathcomponents[depth:] = [treeline]
			    thissrcpath = string.join([srcpath] + pathcomponents[1:], "/")
			    thisdstpath = string.join([path] + pathcomponents[1:], "/")
			    newlump = Lump()
			    newlump.sethdr("Node-path", thisdstpath)
			    newlump.sethdr("Node-action", "add")
			    props = os.popen("svnlook pl -r%d %s %s" % \
			    (srcrev, quote(repos), quote(thissrcpath)), "r")
			    while 1:
				propname = props.readline()
				if propname == "": break
				if propname[-1:] == "\n": propname = propname[:-1]
				while propname[:1] == " ": propname = propname[1:]
				propf = os.popen("svnlook pg -r%d %s %s %s" % \
				(srcrev, quote(repos), quote(propname), quote(thissrcpath)), "r")
				proptext = propf.read()
				propf.close()
				newlump.setprop(propname, proptext)
			    props.close()
			    if subdir:
				newlump.sethdr("Node-kind", "dir")
			    else:
				newlump.sethdr("Node-kind", "file")
				f = os.popen("svnlook cat -r%d %s %s" % \
				(srcrev, quote(repos), quote(thissrcpath)), "r")
				newlump.text = f.read()
				f.close()
			    contents2.append(newlump)
			tree.close()
			need = 0 # we have now done something
	    if need:
		contents2.append(lump)

	# Change the contents array.
	contents[:] = contents2

	# If we've just removed everything in this revision, leave
	# out some revision properties as well.
	if (len(contents) == 0):
	    revhdr.delprop("svn:log")
	    revhdr.delprop("svn:author")
	    revhdr.delprop("svn:date")

fr = sys.stdin
fw = sys.stdout

repos = sys.argv[1]
paths = InterestingPaths(sys.argv[2:])
print "paths... %s " % paths

# Pass the dump-file header through unchanged.
lump = read_lump(fr)
while not lump.hdrdict.has_key("Revision-number"):
    write_lump(fw, lump)
    lump = read_lump(fr)

revhdr = lump

filt = Filter(paths)

while revhdr != None:
    # Read revision header.
    assert revhdr.hdrdict.has_key("Revision-number")
    contents = []
    # Read revision contents.
    while 1:
	lump = read_lump(fr)
	if lump == None or lump.hdrdict.has_key("Revision-number"):
	    newrevhdr = lump
	    break
	contents.append(lump)

    # Alter the contents of the revision.
    filt.tweak(revhdr, contents)

    # Write out revision.
    write_lump(fw, revhdr)
    for lump in contents:
	write_lump(fw, lump)

    # And loop round again.
    revhdr = newrevhdr

fr.close()
fw.close()
