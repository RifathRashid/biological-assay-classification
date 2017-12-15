#!/bin/sh

cd /home/joycek/cs221_tox/biological-assay-classification

IFS=$'\n'       # make newlines the only separator
let job=0
for line in $(cat hyperparameter_tuning_script); do
	cat > tuning_scripts/"$job"_tune.sh <<EOF 
	$line
EOF
	#submit the job!
	qsub -cwd -N tune_"$job" -l h_vmem=6G -l h_rt=6:00:00 -V -e tuning_scripts/"$job"_error -o tuning_scripts/"$job"_log -A bhatt tuning_scripts/"$job"_tune.sh

	let job++
done