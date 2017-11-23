import sys
from pubchempy import *

def write_to_output(input_file, output_file):
	for index, line in enumerate(input_file):
		parsed_line = line.split('\t')
		smile = parsed_line[0]
		value = parsed_line[2]
		print index
		try: compounds = get_compounds(smile, 'smiles')
		except:
			print "Bad Request"
  			continue
		compound_id = compounds[0].cid
		if compound_id == None: continue
		compound = Compound.from_cid(compound_id)
		fingerprint = compound.cactvs_fingerprint
		output_file.write(str(fingerprint) + '\t' + str(value))

def main():
	input_filename = sys.argv[1]
	input_file = open(input_filename, 'r')
	output_filename = input_filename.strip('.smiles') + '.csv'
	output_file = open(output_filename, 'w')
	write_to_output(input_file, output_file)
	input_file.close()
	output_file.close()

main()