def read_file(filename):
    '''
    Parameters
    - filename: str
        File must be tab-delimited as follows: smiles code, tox21_id, label, fingerprint
    
    Returns
    - (X, Y): tuple of np.arrays
        X is an array of features
        Y is a vector of labels
    '''
    X = []
    Y = []
    input_file = open(filename, 'r')
    
    for index, line in enumerate(input_file):
        
        #input file is in format SMILES code, cid, 881-bit fingerprint, 33 extra features (tab-delimited), y
        
        
        # split line (1 data point) into smiles, fingerprint (features), 33 extra featues, and label
        split_line = line.strip().split('\t')
        fingerprint = [int(c) for c in split_line[2]]
        label = int(split_line[36])
        extra_features = split_line[3:36]
        all_features = fingerprint.extend(extra_features)
        
        # append data point to train_x (features) and train_y (labels)
        X.append(all_features)
        Y.append(label)
    input_file.close()
    return (np.array(X), np.array(Y))
