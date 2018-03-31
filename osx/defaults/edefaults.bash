edefaults-source(){   echo $BASH_SOURCE ; }
edefaults-vi(){       vi $(edefaults-source) ; }
edefaults-svi(){      sudo vi $(edefaults-source) ; }
edefaults-env(){      elocal- ; }
edefaults-usage(){ cat << EOU

* https://github.com/pawelgrzybek/dotfiles/blob/master/setup-macos.sh

* https://github.com/mathiasbynens/dotfiles/blob/master/.macos

EOU
}

edefaults--()
{
# System Preferences > Dock > Automatically hide and show the Dock:
defaults write com.apple.dock autohide -bool true
   
# System Preferences > Trackpad > Tap to click
#defaults write com.apple.driver.AppleBluetoothMultitouch.trackpad Clicking -bool true

defaults write com.apple.AppleMultitouchTrackpad Clicking -bool true
defaults write com.apple.AppleMultitouchTrackpad DragLock -bool true
defaults write com.apple.AppleMultitouchTrackpad Dragging -bool true

defaults write com.apple.universalaccess virtualKeyboardOnOff -bool true

/usr/libexec/PlistBuddy -c "Add :Window\ Settings:Basic:shellExitAction bool 1" ~/Library/Preferences/com.apple.Terminal.plist

}




clr(){  rm -f /tmp/bef /tmp/aft ; }
bef(){  clr   ; defaults read > /tmp/bef ; }
aft(){          defaults read > /tmp/aft ; chg ; } 
chg(){  vimdiff /tmp/bef /tmp/aft ; }

