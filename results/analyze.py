import os

def main():
    best_auc = 0 
    best_point = ''
    for i in xrange(257):
    	if os.path.isfile(str(i) + '.results'):
    		f = open(str(i) + '.results', 'r')
    		lines = f.readlines()
    		line = lines[1].split(',') #get second line in file
    		if line[2] > best_auc:
    			best_auc = line[2]
    			best_point = lines[1]
	print best_point


if __name__ == "__main__":
    main()
