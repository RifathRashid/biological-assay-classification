
# coding: utf-8

# In[13]:


import pubchempy as pcp
import pandas as pd
import argparse
import os


# In[46]:


def get_compound_from_smiles(smiles):
    attempts = 5
    time_delay = 1 # in seconds
    while attempts >= 1:
        try:
            compounds = pcp.get_compounds(smiles, namespace='smiles')
            cid = compounds[0].cid
            if cid == None:
                print('No PubChem record') # https://pubchempy.readthedocs.io/en/latest/guide/gettingstarted.html
                return None
            compound = pcp.Compound.from_cid(cid)
        except:
            attempts -= 1
            print('Could not get compound. ' + str(attempts) + ' attempts remaining.')
            time.sleep(time_delay)
        else:
            return compound
    
    if attempts <= 0:
        print('Failed to get compound from smiles after exhausting all attempts')
        return None

def compound_series(smiles_file):
    compound_series_list = []
    for index, line in enumerate(smiles_file):
        # parse line
        parsed_line = line.strip().split('\t')
        smiles, ncats_id, label = tuple(parsed_line)
        
        # get compound
        print('Getting compound ' + str(index))
        compound = get_compound_from_smiles(smiles)
        
        # check for empty compound (e.g. failed to get compound from smiles code)
        if compound == None:
            continue
        
        # convert compound to pandas Series
        compound_series = compound.to_series()
        
        # append label, smiles, and ncats_id to pandas Series
        compound_series['label'] = label
        compound_series['smiles'] = smiles
        compound_series['ncats_id'] = ncats_id
        
        # add Series to list of compound Series
        compound_series_list.append(compound_series)
    
    return compound_series_list

def compound_series_to_dataframe(compound_series_list):
    df = pd.concat(compound_series_list, axis=1)
    df = df.T
    # df = df.set_index('cid')
    return df


# In[45]:


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str)
    args = parser.parse_args()
    input_filename = args.input_file
    input_directory = 'data/'
    output_directory = 'features-pandas/'
    output_filename = output_directory + input_filename + '.features'

    with open('data/'+ input_filename + ".smiles", 'r') as smiles_file:
        compound_series_list = compound_series(smiles_file)

    df = compound_series_to_dataframe(compound_series_list)
    df.to_csv(output_filename, sep='\t')

if __name__ == "__main__":
    main()

