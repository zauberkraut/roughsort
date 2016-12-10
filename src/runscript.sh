#!/bin/bash
for k in 3 50 300 5000
	do
		echo $k >> res.out
		echo "---------------------" >> res.out
		echo "" > resfile
		for n in 8192 16384 32768 65536 131072 262144 524288 1048576 2097152 4194304 8388608 16777216
		do
			for i in {1..5}
			do
				echo "$n `./roughsort.exe -m -s -n $n -k $k`" >> resfile
			done
			cat resfile | awk "{sum+=\$2;sum2+=\$4} END { print \$1 \" \" sum / 5 \" \" sum2 / 5;}" >> res.out
			echo "" > resfile
		done
	done
