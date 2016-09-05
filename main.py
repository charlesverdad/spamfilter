import email
import os
import re
import sys
import numpy as np 		#numpy has smaller memory footprint than python lists.
from multiprocessing import Pool
import itertools
import time
import json  

from settings import *


currPath =  os.path.dirname(os.path.realpath(__file__))

#isSpam = {}
#vocab = {}
#supported_content_types = ['text/plain', 'text/html', 'multipart', 'multipart/alternative']
def dprint(msg):
	if DEBUG:
		print(msg)
def vprint(msg):
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
	dprint ("Total Dataset: " + str(len(trainingPaths) + len(testingPaths)))
	dprint (str(len(trainingPaths)) + " Training Documents")
	dprint (str(len(testingPaths)) + " Testing Documents")

	return (trainingPaths, testingPaths)

def parseWords(msg):
	#given string of message, return list of valid words.
	output = set()
	cleaned = cleanMessage(msg)
	words = re.findall('[a-zA-z]+', cleaned)
	
	for word in words:
		if CASE_INSENSITIVE: word = word.lower()
		if len(word) < 20: output.add(word)
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
		'''		p = Pool(processes=4)
		results = p.map_async(getWords, trainingPaths)
		p.close() # No more getWords
		while (True):
			if (results.ready()): break
			remaining = results._number_left
			print ("\rRemaining tasks: " + str(remaining)),
			time.sleep(0.5)'''
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

def generateWordArray_star(a_b):
	#wrapper for use in multiprocessing
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

	return 1.0 * (sums+addend) / docsInClass

def computeProbabilities(Xspam, Xham, vocabmap, Y):
	
	totalSpam = Xspam.shape[0] 		#total number of spam documents in training set
	totalHam = Xham.shape[0]		#total number of ham documents in training set
	totalDocs = totalSpam + totalHam

	vprint("Pre-computing Priors...")
	P_w_is_S = 1.0 * totalSpam / totalDocs
	P_w_is_H = 1.0 * totalHam / totalDocs
	vprint("Prior Computed.")

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

def classifyEmail(wordArray, P_w_is_S, P_w_is_H, P_x_gvn_S, P_x_gvn_H, Y):
	#given the word Array, probabilities and the lambda, returns True if Spam.
	 
def main():

	trainingPaths, testingPaths = generatePaths()			#generate list of paths for training and testing
	isSpam = parseLabels(pathLabels)			#
	

	if not LOAD_VOCAB:
		vocab = generateVocabulary(trainingPaths)
		#build a hashmap for lookup of the index of the word in the vocabulary
		vocabmap = {}
		for index, word in enumerate(vocab):					
			vocabmap[word] = index
		saveData(vocabmap, 'vocabmap')
	else:
		vocabmap = loadData('Output/vocabmap.json')

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

		P_w_is_S, P_w_is_H, P_x_gvn_S, P_x_gvn_H = computeProbabilities(Xspam, Xham, vocabmap, 0)
if __name__ == '__main__':
	main()