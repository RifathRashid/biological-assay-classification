Modeling and prediction of drug toxicity from chemical structure
By Joyce Kang, Rifath Rashid, Benjamin Yeh

DATA:

data_pcfp: For each allocation of train, test, and score: contains the fingerprints for samples of each assay

data_pcfp: For each allocation of train, test, and score: cotnains the fingerprints for samples and additional PubChem properties of each assay

data_pcfp: For each allocation of train, test, and score: contains fingerprints and additional properties as dataframes of each assay

data_raw:  For each allocation of train, test, and score: contains SMILE codes for samples of each assay

CODE: 

p-progress: contains code used to confer SMILES to fingerprints and basic naive_bayes classifier (used for progress report)

results: contains results of various metrics for different combinations of parameters

results_score: output directory for results of a given run

DNN-tensorflow-manual.py: generates neural network through tensorflow and outputs relevant metrics based on parameters entered from the command line

DNN-tensorflow-tuning.py: generates neural network through tensorflow but doesn't output plots 

hyperparameter_tuning_script: shell script to run combinations of different parameters

new_read_file_method.py: method to parse file of training inputs

smile_to_features2.py: method to convert SMILES code to fingerprints via the PubChem API

smiles_to_features-pandas.py: takes features and converts them into a pandas dataframe for more efficient processing

submit_jobs.sh: shell script to run multiple jobs in parallel in the genomic cluster

COMMANDS: 

Example command with best accuracy: 

"python DNN-tensorflow-tuning.py --run_id 57 --rand_seed 848 --assay_name nr-ahr --data_dir data_pcfp_ext --data_file_ext features --loss_balance --kernel_reg_const 0.1 --batch_size 100 --num_epochs 4 --node_array 512"



