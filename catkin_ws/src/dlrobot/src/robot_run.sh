#!/bin/bash

# Launch openni2 for Xtion Pro camera
roslaunch openni2_launch openni2.launch > openni2.log 2>&1 &
openni2pid=$!
echo "Starting openni2..."
sleep 5

# reconfigure the camera resolution
# NOT ALL modes are supported - mode 5,8 and 11 are supported in color mode!
rosrun dynamic_reconfigure dynparam set /camera/driver  "{'color_mode':'8', 'depth_mode':'8', 'ir_mode':'8'}" 2>&1 &

# PS 3 controller (Turn on bluetooth)
#echo 'ubuntu' | sudo -S sixad -s &
#sixpid=$1 # Save the PID so we can kill it later

# launch kobuki node
roslaunch kobuki_node minimal.launch  > kobuki.log 2>&1 &
kobukipid=$!


# function called by trap
ctrl_c() {
	echo "ctrl+c pressed!"
	# Clean up and kill the processes
	kill $openni2pid
	kill $kobukipid
	#kill $sixpid
	exit
}

# trap ctrl-c and call ctrl_c()
trap 'ctrl_c' SIGINT

input="$@"
while true; do
	echo "Press ctrl+c close the program"
	read input
done

kill $openni2pid
kill $kobukipid
#kill $sixpid
