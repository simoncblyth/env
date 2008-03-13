

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
