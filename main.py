import email
import os
import re
import sys
import numpy as np 		#numpy has smaller memory footprint than python lists.
from multiprocessing import Pool
import itertools
import time
import json 
import math 

from settings import *

np.set_printoptions(threshold = np.inf)
old_settings = np.seterr(all='raise')
currPath =  os.path.dirname(os.path.realpath(__file__))

outputcontent = ''
#isSpam = {}
#vocab = {}
#supported_content_types = ['text/plain', 'text/html', 'multipart', 'multipart/alternative']
def dprint(msg):
	if DEBUG:
		print(msg)
def vprint(msg):
	global outputcontent
	outputcontent += str(msg) + '\n'
	if VERBOSE:
		print(msg)

def removeBase64(raw):
	#catch stray base64 encoded stuff and remove them
	#base64pattern ='^([A-Za-z0-9+/]{4})*([A-Za-z0-9+/]{4}|[A-Za-z0-9+/]{3}=|[A-Za-z0-9+/]{2}==)(\r|\n)$' #doesn't work for some reason
	pattern = '(([A-Za-z0-9+/]{4}){18})'		#only removes the long ones: 4 * 18 characters.
	return re.sub(pattern, '', raw)

def cleanMessage(raw_html):	
	tmp = removeBase64(raw_html)
	tmp = re.sub('\n', ' ', tmp)	
	if REMOVE_CSS: tmp = re.sub('<style>.*<\/style>', '', tmp)				#remove all css marked by <style> tags
	if REMOVE_HTML: tmp = re.sub('<[^>]*>','', tmp)							#strip all html tags including nested tags

	return re.sub('\&.*\;', '', tmp)				#strip all special html characters like &nbsp;

def getEmailBody(fn):
	with open(fn) as f:
		mail = email.message_from_file(f)
	output = ''
	for part in email.iterators.typed_subpart_iterator(mail):				#iterates on subparts that have Content-type text/.*
																			#ignore other content types such as application/ms-tnef image/png etc
		dprint(part.get_content_type())
		for msg in email.iterators.body_line_iterator(part, decode=True):		
			output += msg.decode(errors='ignore')
	return output

def generatePaths():
	#generate list of paths for training and testing
	trainingPaths = []
	testingPaths = []
	datasetPath = os.path.join(currPath, datasetFolder)
	for (dirpath, dirnames, filenames) in os.walk(datasetPath):
		try:
			if int(dirpath.split('/')[-1]) <= TRAINING_BOUNDARY:
				trainingPaths.extend([dirpath + '/' + x for x in filenames])
			else:
				testingPaths.extend([dirpath + '/' + x for x in filenames])
		except ValueError:
			pass
	vprint ("Total Dataset: " + str(len(trainingPaths) + len(testingPaths)))
	vprint (str(len(trainingPaths)) + " Training Documents")
	vprint (str(len(testingPaths)) + " Testing Documents")

	return (trainingPaths, testingPaths)

def parseWords(msg):
	#given string of message, return list of valid words.
	output = set()
	cleaned = cleanMessage(msg)
	words = re.findall('[a-zA-z]+', cleaned)
	
	for word in words:
		if CASE_INSENSITIVE: word = word.lower()
		if len(word) < 20: output.add(word)

	#print output
	return output

def getWords(path):
	dprint("current progress: " + path)
	return parseWords(getEmailBody(path))

def parseLabels(fn):
	#parse labels file and outputs an 'isSpam' dictionary with filename '000/000' as key
	output = {}

	with open(fn) as f:
		doc = f.read()
	lines = doc.split('\n')
	
	for line in lines:
		tmp = line.split(' ')
		if len(tmp) != 2: continue #ignore empty line
		w, path = tmp[0], tmp[1][-7:]
		if w == 'spam':
			output[path] = True
		else:
			output[path] = False

	return output


def generateVocabulary(trainingPaths):
	vprint("Generating Vocabulary...")
	if MULTIPROCESS:
		pool = Pool(processes=PROCESSES)
		results = pool.map(getWords, trainingPaths)
	else:
		results= map(getWords, trainingPaths)

	dictionary = set()
	for result in results:
		dictionary.update(result)		#unite the global dictionary with the set from the results

	vprint("Done. " + str(len(dictionary)) + " words.")
	return dictionary

def generateWordArray(path, vocabmap):
	#access the email file in path and generate a boolean array indicating the presence of the word.
	dprint ("in generateWordArray() " + path)
	output = np.zeros((1, len(vocabmap)), dtype=bool)		#initialize output to array of False (0) wih length equal to number of words in vocab
	#vocablist = list(vocab)

	for word in getWords(path):			
		if word in vocabmap:
			output[0][vocabmap[word]] = True 	#assert the corresponding index of the word in the dictionary

	return output

def generateWordIndexList(path, vocabmap):
	#generate a list of indices on vocabmap of the words contained in the email.
	#basically this is just like getting the indices of the true values of the output of generateWordArray()
	#It is much faster to use this in testing than generateWordArray() because most words are not present.
	dprint ("in generateWordArray() " + path)
	output = []
	#vocablist = list(vocab)

	for word in getWords(path):			
		if word in vocabmap:
			output.append(vocabmap[word])
	return output

def generateWordArray_star(a_b):
	#wrapper for using generateWordArray in multiprocessing
	#convert func([a,b]) to func(a,b) call
	return generateWordArray(*a_b)

def buildTrainingMatrix(paths, vocabmap):
	matrix_sequences = []
	if MULTIPROCESS:								#http://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments
		pool = Pool(processes=PROCESSES)
		results = pool.map(generateWordArray_star, itertools.izip(paths, itertools.repeat(vocabmap)))
		matrix_sequences = results
	else:
		for path in paths:
			matrix_sequences.append(generateWordArray(path, vocabmap))

	return matrix_sequences


def buildTrainingMatrices(trainingPaths, isSpam, vocabmap):
	#returns the matrices Xspam and Xham
	# Xspam[0] accesses the document wordarray
	# Xspam[0][12] returns true when the word in index 12 exists in document 0.
	# in multiprocessing mode, the documents are unordered.
	spamPaths = []
	hamPaths = []

	#segregate training paths into spam and ham
	for path in trainingPaths:
		if isSpam[path[-7:]]:
			spamPaths.append(path)
		else:
			hamPaths.append(path)

	print(len(spamPaths), len(hamPaths))

	vprint("Started building Xspam...")
	xspam_sequences = buildTrainingMatrix(spamPaths, vocabmap)
	Xspam = np.concatenate(xspam_sequences)
	vprint("Finished building Xspam. size: " + str(Xspam.shape))

	vprint("Started building Xham...")
	xham_sequences = buildTrainingMatrix(hamPaths, vocabmap)
	Xham = np.concatenate(xham_sequences)
	vprint("Finished building Xham. size: " + str(Xham.shape))
 
	return Xspam, Xham

def computeLikelihood(X, vocabmap, Y):
	#Y is the lambda parameter for Lambda smoothing
	docsInClass = X.shape[0]	#total number of spam or ham documents in the training set
	sums = np.sum(X, axis=0)

	addend = np.ones(len(vocabmap)) * Y
	output =  1.0 * (sums+addend) / (docsInClass + (Y * len(vocabmap)))

	return output

def computeProbabilities(Xspam, Xham, vocabmap, Y):
	
	totalSpam = Xspam.shape[0] 		#total number of spam documents in training set
	totalHam = Xham.shape[0]		#total number of ham documents in training set
	totalDocs = totalSpam + totalHam

	vprint("Pre-computing Priors...")
	P_w_is_S = 1.0 * totalSpam / totalDocs
	P_w_is_H = 1.0 * totalHam / totalDocs
	vprint("Priors: P(S)="+str(P_w_is_S) + " P(H)=" + str(P_w_is_H))

	vprint("Pre-computing Likelihoods...")
	P_x_gvn_S = computeLikelihood(Xspam, vocabmap,Y)
	P_x_gvn_H = computeLikelihood(Xham, vocabmap,Y)
	vprint("Likelihoods computed.")

	return (P_w_is_S, P_w_is_H, P_x_gvn_S, P_x_gvn_H)

def saveData(data, filename):
	vprint("Saving " + filename + "...")
	path = os.path.join(outputFolder, filename)
	
	if type(data).__name__ == 'ndarray':
		path += '.npy'
		np.save(path, data)
	else:
		path += '.json'
		with open(path, 'w') as f:
			json.dump(data, f)

	vprint("Data Saved: " + path)

def loadData(filename):
	if not os.path.isfile(filename):
		sys.exit("The file " + filename + "is not found and cannot be loaded. Compute the data first by changing LOAD variables in settings.py") 
	with open(filename) as f:
		data = json.load(f)
	vprint("Loaded " + filename)
	return data

def switchTerm(switch, value):
	return value if switch else (1-value)
def switchTerm_star(a_b):
	return switchTerm(*a_b)

def classifyEmail(wordIndexList, P_w_is_S, P_w_is_H, P_x_gvn_S, P_x_gvn_H):
	#given the word Array, probabilities and the lambda, returns True if Spam.
	#wordArray = wordArray[0]


	#S_terms = np.ones(len(wordArray))
	#H_terms = np.ones(len(wordArray))
	'''for i in range(len(wordArray)):
					if (wordArray[i]):
						S_terms[i] = P_x_gvn_S[i]
						H_terms[i] = P_x_gvn_H[i]
					else:
						S_terms[i] = 1 - P_x_gvn_S[i]
						H_terms[i] = 1 - P_x_gvn_H[i]'''

	#list comprehensions should be faster
	#S_terms = np.array([P_x_gvn_S[i] if wordArray[i] else (1 - P_x_gvn_S[i]) for i in range(len(wordArray))])
	#H_terms = np.array([P_x_gvn_H[i] if wordArray[i] else (1 - P_x_gvn_H[i]) for i in range(len(wordArray))])

	#cannot do multiprocessing inside a subprocess
	'''pool = Pool(processes = 4)
				S_terms = pool.map(switchTerm_star, itertools.izip(wordArray, P_x_gvn_S))
				H_terms = pool.map(switchTerm_star, itertools.izip(wordArray, P_x_gvn_H))
	'''

	S_terms = 1 - P_x_gvn_S
	H_terms = 1 - P_x_gvn_H
	for i in wordIndexList:
		S_terms[i] = P_x_gvn_S[i]
		H_terms[i] = P_x_gvn_H[i]

	partial_S = np.sum(np.log(S_terms)) + math.log(P_w_is_S)
	partial_H = np.sum(np.log(H_terms)) + math.log(P_w_is_H)
	prob_S = partial_S# - (partial_S + partial_H)  #the formula is subtraction, not division, in log space ?
	prob_H = partial_H# - (partial_S + partial_H)
	#prob_S = math.exp(partial_S) / (math.exp(partial_S) + math.exp(partial_H))
	#prob_H = math.exp(partial_H) / (math.exp(partial_S) + math.exp(partial_H))

	if prob_S > prob_H:
		return True
	else:
		return False

def getStat(path, vocabmap,  P_w_is_S, P_w_is_H, P_x_gvn_S, P_x_gvn_H, isSpam ):
	#returns a tuple containing a value(TP, TN, FP, FN)
	#e.g. for true positive (1,0,0,0)
	
	result = classifyEmail(generateWordIndexList(path, vocabmap), P_w_is_S, P_w_is_H, P_x_gvn_S, P_x_gvn_H)
	actual = isSpam[path[-7:]]

	case = (result, actual)
	if case == (1,1):
		return (1,0,0,0)
	elif case == (0,0):
		return (0,1,0,0)
	elif case == (1,0):
		return (0,0,1,0)
	elif case == (0,1):
		return (0,0,0,1)

def getStat_star(a_b):
	#wrapper function of getStat(). used for multiprocessing with more than one argument
	return getStat(*a_b)

def getStats(paths, vocabmap, Xspam, Xham, isSpam, Y):

	P_w_is_S, P_w_is_H, P_x_gvn_S, P_x_gvn_H = computeProbabilities(Xspam, Xham, vocabmap, Y)
	vprint("Classifying Emails...")
	results = []
	if MULTIPROCESS:
		pool = Pool(processes = PROCESSES)
		it = lambda x: itertools.repeat(x)		#make a wrapper it(x) for itertools.repeat(x)
		results = pool.map(getStat_star, itertools.izip(paths, it(vocabmap), it(P_w_is_S), it(P_w_is_H), it(P_x_gvn_S), it(P_x_gvn_H), it(isSpam)))
	else:
		for path in paths:
			result = getStat(path, vocabmap, P_w_is_S, P_w_is_H, P_x_gvn_S, P_x_gvn_H, isSpam)
			results.append(result)
			print result

	tmp = np.vstack(results)			#concatenate all results
	stats = np.sum(tmp, axis=0)		#get column sum of the results. this will be an array [TP, TN, FP, FN]

	return stats

def getTop200Words(vocablist, vocabmap, Xspam, Xham):
	#returns a list of the top 200 most influencial words
	spams = np.sum(Xspam, axis=0)	#returns a numpy array of all the 
	hams = np.sum(Xham, axis=0)
	difference = np.abs(spams-hams)
	top200 = []
	sortedindices = difference.argsort();
	for i in reversed(range(len(sortedindices)-200, len(sortedindices))):
		top200.append(sortedindices[i])

	for i in top200:
		print vocablist[i], [i],spams[i], hams[i]

	return top200

def getInfrequentWords(vocablist, vocabmap, Xspam, Xham, threshold=3):
	#returns the set of words that occured less than the threshold
	output = set()
	sums = np.sum(Xspam, axis=0) + np.sum(Xham, axis=0)
	
	for i in range(len(sums)):
		if sums[i] <= threshold:
			output.add(vocablist[i])
	
	return output


def main():

	trainingPaths, testingPaths = generatePaths()			#generate list of paths for training and testing
	isSpam = parseLabels(pathLabels)			#
	

	if not LOAD_VOCAB:
		vocab = generateVocabulary(trainingPaths)
		#build a hashmap for lookup of the index of the word in the vocabulary
		vocabmap = {}
		for index, word in enumerate(vocab):					
			vocabmap[word] = index
		if SAVE_VOCAB:
			saveData(vocabmap, 'vocabmap')
			saveData(list(vocab), 'vocablist')
	else:
		vocabmap = loadData('Output/vocabmap.json')
		vocab = set(loadData('Output/vocablist.json'))

	if not LOAD_X_DATA:
		Xspam, Xham = buildTrainingMatrices(trainingPaths, isSpam, vocabmap)
		if SAVE_X_DATA:		
			saveData(Xspam, 'Xspam')
			saveData(Xham, 'Xham')
	else:
		vprint("Loading X datas...")
		Xspam = np.load('Output/Xspam.npy')
		Xham = np.load('Output/Xham.npy')
		vprint("Loaded Xspam and Xham")

	#infrequent = getInfrequentWords(list(vocab), vocabmap, Xspam, Xham)
	#print vocab - infrequent


	#top200 = getTop200Words(list(vocab), vocabmap, Xspam, Xham)
	#for i in top200:
	#	print list(vocab)[i] + " " + str(np.sum(Xspam, axis=0)[i]) + " " + str(np.sum(Xham, axis=0)[i])

	lambdas = [2.0, 1.0, 0.5, 0.1, 0.005]
	stats = {}
	for y in lambdas:
		vprint ("---\n\nComputing stats for lambda: " + str(y))
		stats[y] = getStats(testingPaths, vocabmap, Xspam, Xham, isSpam, y)
		tmp = stats[y]
		vprint("stats for lambda = " + str(y) + " " + str(stats[y]))
		vprint("precision = " + str(float(tmp[0]) / (tmp[0] + tmp[2]) ))
		vprint("recall = " + str(float(tmp[0]) / (tmp[0] + tmp[3])))
		vprint("accuracy = " + str(float(tmp[0]+tmp[1])/(tmp[2]+tmp[3])))

	print "---\n\n"
	print stats
	saveData(str(stats), 'Stats')

	for key in stats.keys():
		stat = stats[key]
		precision = float(stat[0]) / (stat[0] + stat[2])
		recall = float(stat[0]) / (stat[0] + stat[3])
		vprint ("lambda: " + str(key) + " precision: " + str(precision) + " recall: " + str(recall))


	with open('Output/printedoutput.txt', 'w') as f:
		f.write(outputcontent)


if __name__ == '__main__':
	main()