import sys
import os
import argparse
import time
from pubchempy import *

def write_to_output(input_file, output_file):
    for index, line in enumerate(input_file):
        parsed_line = line.split('\t')
        smile = parsed_line[0]
        value = parsed_line[2]
        print('Fingerprinting compound ' + str(index))
        
        attempts = 5 # 5 attempts
        while attempts >= 1:
            try:
                compounds = get_compounds(smile, 'smiles')
            except:
                attempts -= 1
                print('Could not get compound from smiles. ' + str(attempts) + ' attempts remaining.')
                time.sleep(2) # delay for 2 seconds
            else:
                break
        if attempts <= 0:
            print('Failed to get compound from smiles after exhausting all attempts')
          
        if len(compounds) == 0: continue
        compound_id = compounds[0].cid
        if compound_id == None: continue
        
        attempts = 5
        while attempts >= 1:
            try:
                compound = Compound.from_cid(compound_id)
            except:
                attempts -= 1
                print('Could not get compound from smiles. ' + str(attempts) + ' attempts remaining.')
                time.sleep(2) # delay for 2 seconds
            else:
                break
        if attempts <= 0:
            print('Failed to get compound from cid after exhausting all attempts')
        
        fingerprint = compound.cactvs_fingerprint
        
        # get other properties of compound
        p = get_properties('MolecularWeight,XLogP,ExactMass,MonoisotopicMass,TPSA,Complexity,Charge,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,HeavyAtomCount,IsotopeAtomCount,AtomStereoCount,DefinedAtomStereoCount,UndefinedAtomStereoCount,BondStereoCount,DefinedBondStereoCount,UndefinedBondStereoCount,CovalentUnitCount,Volume3D,XStericQuadrupole3D,YStericQuadrupole3D,ZStericQuadrupole3D,FeatureCount3D,FeatureAcceptorCount3D,FeatureDonorCount3D,FeatureAnionCount3D,FeatureCationCount3D,FeatureRingCount3D,FeatureHydrophobeCount3D,ConformerModelRMSD3D,EffectiveRotorCount3D,ConformerCount3D', str(compound_id), 'cid')
        properties_string = convert_p_to_string(p[0])
        
        if len(p[0]) < 33:
        	continue
        
        # format of output file: smile compound_id    fingerprint    properties y-value
        output_file.write(str(smile) + '\t' +  str(compound_id) + '\t' +  str(fingerprint) + '\t' + properties_string + str(value))

def convert_p_to_string(p):
    string = ''
    for key in sorted(p.keys()):
    	if key == 'CID': continue
        string += (str(p[key]) + '\t')
    return string

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    args = parser.parse_args()
    input_filename = args.input_file

    input_file = open('data/'+ input_filename + ".smiles", 'r')
    output_filename = 'features2/' + input_filename + '.features'
    output_file = open(output_filename, 'w')
    write_to_output(input_file, output_file)
    input_file.close()
    output_file.close()

if __name__ == "__main__":
    main()
