#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 22:38:02 2017

@author: student
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 20:42:49 2017

@author: student
"""
import codecs
import os
import numpy as np
import matplotlib.pyplot as plt
#def getstats(temp_results='../../temp_results',raw_file_name='annotated_news_text.txt',n_words=20):

    
temp_results='../../temp_results'
raw_file_name='annotated_news_text.txt'
file_path = os.path.join(temp_results,raw_file_name)
textfile = codecs.open(file_path, "r", "utf-8")   

article_word_freq = {}
headline_word_freq ={}

ratio = 0
article_len = 0
headline_len = 0
count_articles=0

#lines = [line for line in textfile]
#first_lines = lines[0:5]

for line in textfile:
    article = line.split('#|#')[1]
    headline = line.split('#|#')[0]
    
    article = article.split(' ')
    headline = headline.split(' ')
    
    if len(article)!=0 and len(headline)!=0:
        count_articles+=1
        
        for words in article:
            if words in article_word_freq:
                article_word_freq[words]+=1
            else:
                article_word_freq[words]=1
        
        for words in headline:
            if words in headline_word_freq:
                headline_word_freq[words]+=1
            else:
                headline_word_freq[words]=1
        
        ratio += len(article)/len(headline)
        article_len += len(article)
        headline_len += len(headline)
    
ratio/=count_articles
article_len/=count_articles
headline_len/=count_articles

print "Total Unique Words in Articles: "+ str(len(article_word_freq))
print "Total Unique Words in Headline:" + str(len(headline_word_freq))
print "Ratio of Article to Headline Length:" + str(ratio)
print "Average Length of Article "+ str(article_len)
print "Average Lentgth of Headline " + str(headline_len)

n_words = 50
article_word_freq_sorted = [(w, article_word_freq[w]) for w in sorted(article_word_freq, key=article_word_freq.get, reverse=True)]
    
most_frequent_words = article_word_freq_sorted[0:n_words]

words = [u" "+word for (word,freq) in most_frequent_words ]
freq =  [freq for (word,freq) in most_frequent_words]  








#import seaborn as sns
#import matplotlib.pyplot as plt
#sns.set_style("whitegrid")
#x = ['Article','Headline']
#y = [article_len,headline_len]
#ax = sns.barplot(x,y)
#ax.axes.set_title('Average Length of Article & Headline', fontsize=16,color="r",alpha=0.5)
#ax.set(ylabel='Average Number of Words')
#fig = ax.get_figure()
#fig.savefig('average_words.png')
#plt.show()







      




#from PIL import Image
#import matplotlib.pyplot as plt
#from wordcloud import WordCloud,ImageColorGenerator
#text=codecs.open(file_path, "r", "utf-8").read()
#
#
#wordcloud = WordCloud(max_font_size=40).generate(text)
#plt.figure()
#plt.imshow(wordcloud, interpolation="bilinear")
#plt.axis("off")
#plt.show()

