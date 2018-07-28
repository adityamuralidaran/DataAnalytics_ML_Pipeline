# Aditya Subramanian Muralidaran
 
'''
1. Enter the actual apiKey.
2. Run in batch to conditions of API hit per day and per minute.
'''

import urllib.request
import urllib.response
import json
from bs4 import BeautifulSoup
import requests


apiKey = '5419e042517b490094580b2e38865bc9'

searchTerm = 'business'
#searchTerm = 'shooting'
#searchTerm = 'gun'

#writeFile1 = open("NYT_Politics_Train.txt", "w", encoding='utf-8')
#writeFile2 = open("NYT_Politics_Test.txt", "w", encoding='utf-8')

for i in range(10):
    filename = ""
    if(i<8):
        filename = "Business_Train_" + str(i)
    else:
        filename = "Business_Test_" + str(i)
    print(i)
    url = 'http://api.nytimes.com/svc/search/v2/articlesearch.json?q='+searchTerm+'&begin_date=20170101&end_date=' \
        '20180501&page='+str(i)+'&fl=web_url&api-key='+apiKey
    response = urllib.request.urlopen(url)
    if (response.getcode() == 200):
        data = json.load(response)
        j = 1
        for doc in data['response']['docs']:
            filename1 = filename + str(j) +".txt"
            writeFile1 = open(filename1, "w", encoding='utf-8')
            print(doc['web_url'])
            html_code = requests.get(doc['web_url'])
            plain_text = html_code.text
            soup = BeautifulSoup(plain_text,'html.parser')
            for para in soup.find_all('p',{'class':'story-content'}):
                writeFile1.write(para.get_text())
                writeFile1.write("\n")
            for para in soup.find_all('p',{'class':'e2kc3sl0'}):
                writeFile1.write(para.get_text())
                writeFile1.write("\n")
            j += 1
            writeFile1.close()