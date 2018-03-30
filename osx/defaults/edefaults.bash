edefaults-source(){   echo $BASH_SOURCE ; }
edefaults-vi(){       vi $(edefaults-source) ; }
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
defaults write com.apple.driver.AppleBluetoothMultitouch.trackpad Clicking -bool true

}
