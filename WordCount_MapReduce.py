# Aditya Subramanian Muralidaran
 
# Run Command: spark-submit WordCount_MapReduce.py Business/ Politics/ Sports/ Tech/ classify_input

from pyspark import SparkConf, SparkContext, RDD
import sys
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import string

#nltk.download('tokenize')
#nltk.download('RegexpTokenizer')
from nltk.tokenize import RegexpTokenizer

#nltk.download('stopwords')
#nltk.download('corpus')
#nltk.download('words')

def preprocess_function(ipdata):
	data_lower = ipdata.lower()
	tokenizer = RegexpTokenizer(r'\w+')
	word_list = tokenizer.tokenize(data_lower)
	return word_list

def word_count_function(word):
	stop_words = nltk.corpus.stopwords.words('english')
	WNL = WordNetLemmatizer()
	custom_words = ['said', 'year', 'one', 'time', 'like', 'would', 'two', 'first', 'day', 'new', 'could', 'also',
			'many', 'back', 'much', 'last', 'get', 'even', 'way', 'make', 'three', 'four', 'come', 'woman',
			'people', 'right', 'may', 'well', 'say', 'part', 'men', 'made']
	root_word = WNL.lemmatize(word)
	if root_word not in stop_words and root_word not in custom_words and root_word.isalpha() and len(root_word)>2:
		return (root_word,1)

def classification_preprocess((file_name,ipdata)):
	data_lower = ipdata.lower()
	tokenizer = RegexpTokenizer(r'\w+')
	WNL = WordNetLemmatizer()
	word_list = tokenizer.tokenize(data_lower)
	list_keyval = list()
	for word in word_list:
		list_keyval.append((file_name,WNL.lemmatize(word)))
	return list_keyval

def feature_sort(val1, val2):
	feature1 = int(str(val1).split(":")[0])
	feature2 = int(str(val2).split(":")[0])
	if(feature1 < feature2):
		return str(val1)+" "+str(val2)
	else:
		return str(val2)+" "+str(val1)

if __name__ == "__main__":
	conf = SparkConf().setAppName("WordCount")
	sc = SparkContext(conf = conf)
	features = set()

	for i in range(1,len(sys.argv)-1,1):
		text_file = sc.textFile(sys.argv[i])
		word_count = text_file.flatMap(preprocess_function).map(word_count_function).filter(lambda x: x is not None).reduceByKey(lambda a, b: a + b)
		#sorting and picking top 20 words as feature for ML prediction
		sorted = word_count.map(lambda (a,b): (b,a)).sortByKey(0).map(lambda (a,b): b).take(40)
		for feature in sorted:
			features.add(feature)

	
	feature_file = sc.parallelize(features)
	feature_file.coalesce(1).saveAsTextFile("features_file")
	print(features)
	#feature_vector = sc.parallelize(features).map(lambda x:x)
	#feature_vector.coalesce(1).saveAsTextFile(sys.argv[len(sys.argv)-1])

	feature_index = dict()
	j = 1
	for feature in features:
		feature_index[feature] = j
		j += 1

	classification_input =  sc.parallelize([])
	for i in range(1, len(sys.argv) - 1, 1):
		article = sc.wholeTextFiles(sys.argv[i])

		#temp1 = article.flatMap(classification_preprocess)
		classification_word_count = article.flatMap(classification_preprocess)\
			.filter(lambda x: x is not None)\
			.map(lambda (file,word): (file,word) if word in feature_index else None) \
			.filter(lambda x: x is not None)\
			.map(lambda (file,word): ((file,feature_index[word]),1))\
			.reduceByKey(lambda a, b: a + b)

		classification_ipformat = classification_word_count\
			.map(lambda (key,val): (str(key[0]), str(key[1])+":"+str(val)))\
			.sortBy(lambda (key,val): int(str(val).split(":")[0]),ascending=True)\
			.reduceByKey(lambda val1, val2: str(val1)+" "+str(val2))\
			.map(lambda (key,val): str(i-1)+" "+str(val))

		classification_input = classification_input.union(classification_ipformat)
		#classification_ipformat.coalesce(1).saveAsTextFile(sys.argv[len(sys.argv) - 1])
		#break

	classification_input.coalesce(1).saveAsTextFile(sys.argv[len(sys.argv) - 1])
	sc.stop()
