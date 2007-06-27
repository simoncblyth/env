
----------------------   AN EASY WAY TO ENTER COMPLEX SETTINGS INTO THE XCODE GUI --------------------

on run argv
	
	set the_target_name to item 1 of argv
	set the_macros to my GetFlagString()
	my SetBuildSetting( the_target_name , "Debug", "GCC_PREPROCESSOR_DEFINITIONS", the_macros)
	
end run



on SetBuildSetting(the_target_name, the_config_name, the_setting_name, the_value)
	-- NB you have to "touch" the setting to smth in the GUI before this will work	
	-- otherwise the setting doesnt exist 
	
	tell application "Xcode"
		set the_target to target the_target_name of project 1
		set the_config to build configuration the_config_name of the_target
		try
			set the_setting to build setting (the_setting_name as text) of the_config
			set value of the_setting to the_value
		on error the_err
			display dialog "NB you must touch the setting in the GUI before this works " & the_err
		end try
	end tell
end SetBuildSetting


on GetFlagString()
	-- flags obtained by cd $G4INSTALL ; ./Configure -cppflags   and looking in the config/*.gmk
	set the_vis to {"DAWNFILE", "HEPREPFILE", "RAYTRACER", "VRMLFILE", "ASCIITREE", "GAGTREE"}
	set the_vis to the_vis & {"DAWN", "OPENGL", "OPENGLX", "OPENGLXM", "OIX", "OI", "RAYTRACERX", "VRML"}
	
	set the_flags to {"G4VERBOSE", "G4_STORE_TRAJECTORY", "G4UI_USE_TCSH", "G4UI_USE_XM"}
	set the_flags to the_flags & {"G4VIS_USE", "G4INTY_USE_XT", "G4ANALYSIS_USE"}
	--
	repeat with i from 1 to count the_vis
		set the_use to "G4VIS_USE_" & (item i of the_vis)
		set the_build to "G4VIS_BUILD_" & (item i of the_vis) & "_DRIVER"
		set the_flags to the_flags & the_build & the_use
	end repeat
	--	
	set the_flag_string to ""
	repeat with i from 1 to count the_flags
		set the_flag_string to the_flag_string & " " & item i of the_flags as text
	end repeat
	return the_flag_string
end GetFlagString

