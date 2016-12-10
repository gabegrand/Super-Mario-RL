#!/bin/bash

fceuxPID=$(ps ax | grep '[/]usr/local/Cellar/fceux' | awk '{print $1}')
kill -9 $fceuxPID