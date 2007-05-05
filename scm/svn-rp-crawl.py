#!/usr/bin/env python 

"""
   Crawl a repository, printing versioned object path names.
   eg 
         svn-rp $SCM_FOLD/repos/red

""" 

import sys 
import os.path 
import svn.fs, svn.core, svn.repos 

def crawl_filesystem_dir(root, directory): 
	"""Recursively crawl DIRECTORY under ROOT in the filesystem, and return a list of all the paths at or below DIRECTORY.""" 
	# Print the name of this path. 
	print directory + "/" 
	# Get the directory entries for DIRECTORY. 
	entries = svn.fs.svn_fs_dir_entries(root, directory) 

	# Loop over the entries. 
	names = entries.keys() 

	for name in names: 
		# Calculate the entry's full path. 
		full_path = directory + '/' + name 

		# If the entry is a directory, recurse. The recursion will return 
		# a list with the entry and all its children, which we will add to 
		# our running list of paths. 
		
		if svn.fs.svn_fs_is_dir(root, full_path): 
			crawl_filesystem_dir(root, full_path) 
		else: 
			# Else it's a file, so print its path here. 
			print full_path 
			
def crawl_youngest(repos_path): 
	"""Open the repository at REPOS_PATH, and recursively crawl its youngest revision.""" 
	# Open the repository at REPOS_PATH, and get a reference to its 
	# versioning filesystem. 
	repos_obj = svn.repos.svn_repos_open(repos_path) 
	fs_obj = svn.repos.svn_repos_fs(repos_obj) 
	# Query the current youngest revision. 
	youngest_rev = svn.fs.svn_fs_youngest_rev(fs_obj) 
	print " youngest_rev %d " % youngest_rev
	# Open a root object representing the youngest (HEAD) revision. 
	root_obj = svn.fs.svn_fs_revision_root(fs_obj, youngest_rev) 
	# Do the recursive crawl. 
	crawl_filesystem_dir(root_obj, "") 
	
if __name__ == "__main__": 
	# Check for sane usage. 
	if len(sys.argv) != 2: 
			sys.stderr.write("Usage: %s REPOS_PATH\n" % (os.path.basename(sys.argv[0]))) 
			sys.exit(1) 
	print sys.argv
	# Canonicalize the repository path. ... removes trailing slashes + ??
	repos_path = svn.core.svn_path_canonicalize(sys.argv[1]) 
	print " repos_path %s " %  repos_path
	# Do the real work. 
	crawl_youngest(repos_path) 


