
authz(){

  local tw="thho, bzhu, wei, adiar, chwang, glin, fengshui, wunsyonghe"
  local hk="jimmy, antony, soap, talent"
  local cn="tianxc, maqm"
  local us="littlejohn"

cat << EOA
#
#   $msg $BASH_SOURCE  $(date)
#
#     tw:[$tw]
#     hk:[$hk]
#     cn:[$cn]
#     us:[$us]
#     
#   http://svnbook.red-bean.com/en/1.0/ch06s04.html
#  
# 
#
[groups]

sync = ntusync
dyuser = blyth, $tw, $hk, $cn, $us, dayabay, slave

evuser = simon, dayabay
evdev = blyth, $tw, $hk, $cn, $us 
evadmin = blyth, dayabaysoft, admin 

abuser = simon, dayabay
abdev = blyth, $tw, $hk
abadmin = blyth

hzuser = simon, cjl, tosi, cecilia, b2c
hzdev = blyth
hzadmin = blyth

tduser = simon
tddev = blyth
tdadmin = blyth

wfuser = simon
wfdev = blyth
wfadmin = blyth


# force authenticated 
[dybsvn:/]
@sync = rw
@dyuser = r 

[env:/]
* = r
@evuser = r
@evdev = rw 
@evadmin = rw

[newtest:/]
* = r
@evuser = r
@evdev = rw 
@evadmin = rw

[data:/]
* = r
@evuser = r
@evdev = rw 
@evadmin = rw

[aberdeen:/]
@abuser = r
@abdev = rw 
@abadmin = rw

[heprez:/]
* = r
@hzuser = rw
@hzdev = rw 
@hzadmin = rw

[tracdev:/]
* = r
@tduser = r
@tddev = rw 
@tdadmin = rw

[workflow:/]
@wfuser = r
@wfdev = rw 
@wfadmin = rw

[ApplicationSupport:/]
@wfuser = r
@wfdev = rw 
@wfadmin = rw
 
EOA


}


