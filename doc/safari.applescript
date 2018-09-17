--
--
--    1:1   1.0 
--    5:4   1.25    
--    4:3   1.333     Standard 4:3: 320x240, 640x480, 800x600, 1024x768
--    8:5   1.6 
--    16:9  1.777     Widescreen 16:9: 640x360, 800x450, 960x540, 1024x576, 1280x720, and 1920x1080
--
--
--   Doubling done by retina screen means cannot directly fit 1920x1080
--   so use same aspect 1280x720
--   or half it to 960x540 ? 
--  

tell application "Safari"

    set width to  1280
    set height to  720

	-- set bounds of window 1 to {0, 22, width, height + 96 + 15}  -- guess the 15 was a bottom status bar that no longer have

	set bounds of window 1 to {0, 22, width, height + 83 }
	get bounds of window 1

    -- workflow:
    --    1. position Safari window flush with top (status bar) and left of screen
    --    2. ensure that there are multiple tabs present 
    --    3. screen capture shift-cmd-4 to get some crosshairs with pixel readout 
    --    3. record coordinates
    --
    --       a. window top left 
    --       b. content top left (not counting the tab as content)
    --       c. content bottom left
    --        
    --    Aim to size window such that y height between b. and c. is precisely 720
    --
    --
    --   High Sierra Safari (2018)
    -- 
    --    {0,22}   window top left
    --    {0,83}   content top left
    --    {0,803}  content bottom left    echo $((803 - 83))  ->  720
    --    {0,803}  window bottom left (no chrome down there)
    --
    --   Unrecorded prior Safari 
    --
    --    {0, 22}   window top left 
    --    {0, 96}   content top left   
    --    {0, 815}  content bottom left  815-96   = 719  
    --    {0, 831}  window bottom left   831 - 815 = 16 
    --    {1280,22} window top right   
    --
	
end tell
