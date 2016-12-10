#!/bin/bash

pyPID=$(ps ax | grep '[p]ython test.py' | awk '{print $1}')
fceuxPID=$(ps ax | grep '[/]usr/local/Cellar/fceux' | awk '{print $1}')
if [[ ( "$pyPID"  =~ ^[0-9]+$ ) || ( "$fceuxPID"  =~ ^[0-9]+$ ) ]]
then
	kill -9 $pyPID
	kill -9 $fceuxPID
	echo "Killed Super Mario Process"
else
	python test.py > out.txt 2>&1 &
    sleep 2
	pyPID=$(ps ax | grep '[p]ython test.py' | awk '{print $1}')
	if [[ "$pyPID"  =~ ^[0-9]+$ ]]
	then
		echo "Launched Super Mario"
	else
		echo 'Failed to launch Super Mario'
	fi
fi
