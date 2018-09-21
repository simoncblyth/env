tell application "Google Chrome"

    set width to  1280
    set height to  720

	set bounds of window 1 to {0, 22, width, height + 96  }
	get bounds of window 1

    --  {0,22}   top left of Chrome window
    --  {0,59}   bottom of tab row
    --  {0,96}   top left of content 
    --  {0,816}  bottom left of content

    --   echo $((816 - 96))   720

end tell
