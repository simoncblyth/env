#!/usr/bin/env python 
"""Crawl a working copy directory, printing status information.""" 
import sys 
import os.path 
import getopt 
import svn.core, svn.client, svn.wc 

def generate_status_code(status): 
	"""Translate a status value into a single-character status code, 
	using the same logic as the Subversion command-line client.""" 
	code_map = { svn.wc.svn_wc_status_none : ' ', 
	svn.wc.svn_wc_status_normal : ' ', 
	svn.wc.svn_wc_status_added : 'A', 
	svn.wc.svn_wc_status_missing : '!', 
	svn.wc.svn_wc_status_incomplete : '!', 
	svn.wc.svn_wc_status_deleted : 'D', 
	svn.wc.svn_wc_status_replaced : 'R', 
	svn.wc.svn_wc_status_modified : 'M', 
	svn.wc.svn_wc_status_merged : 'G', 
	svn.wc.svn_wc_status_conflicted : 'C', 
	svn.wc.svn_wc_status_obstructed : '~', 
	svn.wc.svn_wc_status_ignored : 'I', 
	svn.wc.svn_wc_status_external : 'X', 
	svn.wc.svn_wc_status_unversioned : '?', 
	} 
	return code_map.get(status, '?') 

def do_status(wc_path, verbose): 
	# Calculate the length of the input working copy path. 
	wc_path_len = len(wc_path) 

	# Build a client context baton. 
	ctx = svn.client.svn_client_ctx_t() 
	
	def _status_callback(path, status, root_path_len=wc_path_len): 
		"""A callback function for svn_client_status.""" 
		# Print the path, minus the bit that overlaps with the root of 
		# the status crawl 
		text_status = generate_status_code(status.text_status) 
		prop_status = generate_status_code(status.prop_status) 
		# for "." was getting truncated first char
		#print '%s%s %s' % (text_status, prop_status, path[wc_path_len + 1:]) 
		print '%s%s %s' % (text_status, prop_status, path ) 
		
	# Do the status crawl, using _status_callback() as our callback function. 	
	svn.client.svn_client_status(wc_path, None, _status_callback, 1, verbose, 0, 0, ctx) 

def usage_and_exit(errorcode): 
	"""Print usage message, and exit with ERRORCODE.""" 
	stream = errorcode and sys.stderr or sys.stdout 
	stream.write("""Usage: %s OPTIONS WC-PATH 
	Options: 
	--help, -h : Show this usage message 
	--verbose, -v : Show all statuses, even uninteresting ones 
	""" % (os.path.basename(sys.argv[0]))) 
	
	sys.exit(errorcode) 


if __name__ == '__main__': 
	print "hello " 
	print sys.argv
	#Parse command-line options. 
	try: 
		opts, args = getopt.getopt(sys.argv[1:], "hv", ["help", "verbose"]) 
	except getopt.GetoptError: 
		print "getopterror"
		usage_and_exit(1) 

	print " lenopts %d  " % len(opts)
	print " lenargs %d " % len(args) 
	verbose = 0 
	for opt, arg in opts: 
		if opt in ("-h", "--help"): 
			usage_and_exit(0) 
		if opt in ("-v", "--verbose"): 
			verbose = 1 
    
	if len(args) != 1: 
		print "lenargs error %d " % len(args)
		usage_and_exit(2) 


	# Canonicalize the repository path. ... issue with "."
	print "args0  %s " % args[0]  
	wc_path = svn.core.svn_path_canonicalize(args[0]) 
	print "wc_path %s " % wc_path  

	# Do the real work. 
	try: 
		do_status(wc_path, verbose) 
	except svn.core.SubversionException, e: 
		sys.stderr.write("Error (%d): %s\n" % (e[1], e[0])) 
		sys.exit(1) 
					

