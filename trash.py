#code that are not used in the final program are kept here.

def parseMessage(msg):
	#given a string,
	#returns a set of unique words
	#decoded = unicode(msg, errors='ignore')						#currently ignoring decode errors like japanese and chinese characters
	stripped = cleanmessage(msg)									#strip all html tags
	trimmed = stripped.replace('\n', ' ')						#replace all \n with whitespace
	words = trimmed.split(' ')									#split spaces

	localWords = set()				

	for word in words:
		word = removeNonAlphanumeric(word)
		if len(word) > 30:
			continue
		if word.isdigit():
			continue
		if word == '':
			continue
		elif word[0] == '$':
			localWords.add('$'.decode())							#count all money as one word
		else:
			localWords.add(word.strip('!?.,\\').lower())	#remove trailing punctuations. FEATURE? Make Lowercase
	
	return localWords

def getWords(filename):
	#given filename of an email file, return set of words
	#words are defined as [a-zA-z] delimited by spaces, period, etc.
	contents = getMessage(filename)

	pattern = re.compile('[a-zA-z]+')
	words = re.findall(pattern, contents)

	return set([x for x in words if (len(x)<21)])

def newgetMessage(fn):
	with open(fn) as f:
		mail = email.message_from_file(f)

	output = ''

	for part in mail.walk():
		print '---'
		print part.get_content_type()
		print part['Content-Transfer-Encoding']

def getMessage(filename):
	#given a string path to an email-formatted textfile,
	#returns a string containing the email body.
	with open(filename) as f:
		mail = email.message_from_file(f)

	output = ""
	if mail.is_multipart():
		dprint('multipart')
		dprint(len(mail.get_payload()))
		for item in mail.get_payload():
			dprint(item.get_content_type() + " " + str(item.get_content_charset()))
			if item.get_content_type().lower() not in supported_content_types:
				continue
			output += " "
			try:
				output += item.get_payload().decode(item.get_content_charset(), errors='ignore')
			except:
				print "---++---"
				print item.get_payload()
				sys.stdout.flush()
				output += item.get_payload()

	else:
		dprint(mail.get_content_type() + " " + str(mail.get_content_charset()))
		try:
			output += mail.get_payload(decode=True).decode(mail.get_content_charset(), errors='ignore')
		except:
			output += str(mail.get_payload())

	return cleanmessage(output)


def removeNonAlphanumeric(msg):
	pattern = re.compile('\W')
	return re.sub(pattern, '', msg)

class Vocab:
	def __init__(self):
		self.data = {}
	def addWord(self, word):
		if word in self.data.keys():
			self.data[word] += 1
		else:
			self.data[word] = 1