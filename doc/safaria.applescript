tell application "Safari"
    activate
    delay 0.1
    tell application "System Events" to key code 0 # a key : has javascript key handler to remove some GUI
end tell
