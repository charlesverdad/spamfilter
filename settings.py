datasetFolder = 'trec06p-cs280/data/'
outputFolder = 'Output/'
pathLabels = 'trec06p-cs280/labels'
filename = 'trec06p-cs280/data/000/000'

DEBUG = False
LOAD_VOCAB = True 			#Load previously computed vocabulary data from output.
LOAD_X_DATA = True 			#Load previously computed Xspam and Xham from output.
SAVE_X_DATA = False 		#Save computed Xspam and Xham. requires ~1.3GB of free disk space
MULTIPROCESS = True

REMOVE_CSS = True
REMOVE_HTML = True
CASE_INSENSITIVE = True
TRAINING_BOUNDARY = 70
VERBOSE = True
PROCESSES = 4