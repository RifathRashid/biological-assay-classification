import sys
import argparse
from pubchempy import *

def write_to_output(input_file, output_file):
	for index, line in enumerate(input_file):
		parsed_line = line.split('\t')
		smile = parsed_line[0]
		value = parsed_line[2]
		print('Fingerprinting compound ' + str(index))
		try: compounds = get_compounds(smile, 'smiles')
		except:
			print("Bad Request")
  			continue
		compound_id = compounds[0].cid
		if compound_id == None: continue
		compound = Compound.from_cid(compound_id)
		fingerprint = compound.cactvs_fingerprint
		output_file.write(str(fingerprint) + '\t' + str(value))

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('input_file', type=str)
	args = parser.parse_args()
	input_filename = args.input_file

	input_file = open(input_filename, 'r')
	output_filename = '.' + input_filename.strip('.smiles') + '.csv'
	print(input_filename)
	print(output_filename)
	output_file = open(output_filename, 'w')
	write_to_output(input_file, output_file)
	input_file.close()
	output_file.close()

if __name__ == "__main__":
	main()