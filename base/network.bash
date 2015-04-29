network-vi(){ vi $BASH_SOURCE ; }

network-usage(){ cat << EOU

Network Issues
===============


OSX : Internet sharing from G4PB to Delta over adhoc WiFi not working 
-----------------------------------------------------------------------

* wifi connection is OK and can ssh from Delta to G4PB but 
  internet sharing not working, curl just hangs

* reboot Delta makes no difference, this is with a just rebooted G4PB

Fix: is to disable/enable "System Preferences > Sharing > Internet sharing" 
     on G4PB


OSX : share WiFi to WiFi ?
----------------------------

* http://superuser.com/questions/233924/share-a-wifi-connection-through-wifi-on-mac-os-x

The below looks liable to cause more problems than it solves::

    sudo -s
    ifconfig en1 x.x.x.1/24 alias
    sysctl -w net.inet.ip.forwarding=1
    natd -interface en1
    ipfw -f flush
    ipfw add divert natd all from any to any via en1
    ipfw add pass all from any to any


Why do that:

* to connect iPod/iPad to MBP disk over AFP it needs to be 
  an AdHoc network coming from the MBP

  * http://www.goodiware.com/gr-man-tr-wifi-create.html



EOU
}

network-test(){
	net=`hostname -s`
	#net="localhost"
}

network-wait(){

     ## wait around for hostname to be something other than localhost for up to 30 seconds

        system-

        network-test
		local savenet=$net
        local cnt=0
        local tot=0

        while [ "X$net" = "X" ] || [ "X$net" = "Xlocalhost"  ]
        do
            # Loop for up to 30 seconds
            if [ "$tot" -lt "30" ]
            then
                if [ "$tot" -lt "5" ]
                then
                    cnt=`expr $cnt + 1`
                else
                    echo "Waiting for network ($net,$cnt,$tot) startup ..."
                    cnt=0
                fi
                tot=`expr $tot + 1`

                sleep 1

                network-test
			else
				savenet=$net
				net="timeout"
            fi
        done
		
		net=$savenet
        local ret=0

        if [ "X$net" = "X" ] || [ "X$net" = "Xlocalhost" ]
        then
            echo "$msg Timed out waiting for network ($net) to start"
            system-log  $msg "Timed out waiting for network ($net) to start"
            ret=0
		else
			echo "$msg Network seems to be ready ($net) "
			system-log $msg "Network seems to be ready ($net) "
            ret=10
		fi
	   return $ret
}
