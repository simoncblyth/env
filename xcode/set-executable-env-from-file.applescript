--
--     this was developed in "Smile" in the file : xcode-scripting-executable-env 
--
--
--    this script allows the environment of an executable in Xcode to be set from the
--    values in a file ... saving lots of tedious typing variables into the Xcode GUI one by one
--
--    NB if the executable info panel is open in the GUI changes to existing environment variables
--   are reflected immediately , BUT additions/deletions of variables are not reflected ...
--  to see the additions/deletions close and reopen the info panel
--
--   see the start of ~/.bash_profile for the bash function to create the
--   file with the environment file 
--
--
--      DYLD_LIBRARY_PATH
--      ROOTSYS
--      DAYA_DATA_DIR    
--      DISPLAY   localhost:0.0
--

-- hmm where are the log messages going when invoked from osascript ???
--  NB can copy "somtheiong" to stdout    BUT only the last thing appears on stdout ???
-- 

on run argv
	tell application "Xcode"
		
		try
			set the_executable_name to item 1 of argv
			set the_project_name to item 2 of argv
			set the_posixpath to item 3 of argv
			log " got arguments name:[" & the_executable_name & "] in project:[" & the_project_name & "] path:[" & the_posixpath & "]"
		on error
			return " error accessing arguments name:[" & the_executable_name & "] in project:[" & the_project_name & "] path:[" & the_posixpath & "]"
		end try
		
		try
			set the_project to project the_project_name
		on error
			return "error finding project [" & the_project_name & "]"
		end try
		
		try
			set the_executable to executable the_executable_name of the_project
		on error
			return " error finding executable of name:[" & the_executable_name & "] in project:[" & the_project_name & "]"
		end try
		
		try
			my set_env_from_file(the_executable, the_posixpath)
			my list_env(the_executable)
			
		on error
			display dialog "Must invoke with osascript from command line with 3 arguments : executable name, project name and path to environment file (env dump format) "
		end try
		
	end tell
end run







on testing()
	tell application "Xcode"
		
		--set the_home to POSIX path of (path to home folder) -- includes a trailing slash 
		--set the_posixpath to the_home & "g4dyb_env.txt"
		
		set the_executable to executable "G4dybApp" of project "dyw"
		set_variable(the_executable, "TEST_NAMEX", "hxmmm", true)
		set x to get_variable(the_executable, "TEST_NAME")
		delete x
	end tell
end testing


on set_env_from_file(the_executable, the_posixpath)
	set the_contents to my read_file(the_posixpath)
	set the_line_list to my split(the_contents, "
")
	repeat with i from 1 to count the_line_list
		set the_line to item i of the_line_list
		set the_pair to my split(the_line, "=")
		
		if ((count of the_pair) ­ 2) then
			log " skipping line  [" & the_line & "] with count  : " & (count of the_pair)
		else
			set the_name to item 1 of the_pair
			set the_value to item 2 of the_pair
			set the_active to true
			set_variable(the_executable, the_name, the_value, the_active)
		end if
	end repeat
end set_env_from_file




on clean_env(the_executable)
	tell application "Xcode"
		set the_env_list to environment variables of the_executable
		repeat with i from 1 to count the_env_list
			set the_var to item i of the_env_list
			delete the_var
		end repeat
	end tell
end clean_env


on list_env(the_executable)
	tell application "Xcode"
		set the_env_list to environment variables of the_executable
		repeat with i from 1 to count the_env_list
			set the_var to item i of the_env_list
			log "active:" & active of the_var & " name:" & name of the_var & " value:" & value of the_var
		end repeat
	end tell
end list_env

on get_variable(the_executable, the_var_name)
	tell application "Xcode"
		try
			set the_var to environment variable the_var_name of the_executable
		on error
			set the_var to missing value
		end try
		return the_var
	end tell
end get_variable

on set_variable(the_executable, the_name, the_value, the_active)
	
	--  if a variable of the_name exists already then just change the_value and the_active 
	--      otherwise create a new variable 
	
	tell application "Xcode"
		set the_var to my get_variable(the_executable, the_name)
		if (the_var is missing value) then
			try
				tell the_executable
					set the_var to make new environment variable with properties {name:the_name, value:the_value, active:the_active}
				end tell
			on error
				log "error when creating new environment variable "
			end try
		else
			set value of the_var to the_value
			set active of the_var to the_active
		end if
		return the_var
	end tell
end set_variable

on read_file(the_posix_path)
	set the_file to POSIX file the_posix_path
	open for access the_file
	set the_contents to (read the_file)
	close access the_file
	return the_contents
end read_file

on split(the_string, the_delim)
	set old_delim to AppleScript's text item delimiters
	set AppleScript's text item delimiters to the_delim
	set the_items to text items of the_string
	set AppleScript's text item delimiters to old_delim
	return the_items
end split
