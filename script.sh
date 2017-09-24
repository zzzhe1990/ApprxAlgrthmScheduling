#!/bin/bash
#This is script is to run all distributions in a loop that each distribution is launched by a single process.

for th in '4'
do
	for val in {1..100}
	do
		./ApproxGPURun 0.3 4 $val $th > ~/Desktop/ApproxAlgorthim/Git/ApprxAlgrthmScheduling/output/GPU_Results_File${val}K40.txt & 
		#PID=$!
		sleep 200
		PROCESS_NUM=$(ps -ef | grep ApproxGPURun | grep -v grep | wc -l)
		PID=$(ps -ef | grep ApproxGPURun | grep -v grep | awk '{print $2}')
		if [ $PROCESS_NUM -ne 0 ]
		then
			kill -15 $PID
			rm ~/Desktop/ApproxAlgorthim/Git/ApprxAlgrthmScheduling/output/GPU_Results_File${val}K40.txt
		fi
	done
done
