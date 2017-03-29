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
raw_file_name='raw_news_annotated.txt'
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
heads = []
arts = []
for line in textfile:
    article = line.split('#|#')[1]
    headline = line.split('#|#')[0]
    
    heads.append(headline)
    arts.append(article)
    
    
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


plt.plot([article_word_freq[w] for w,v in article_word_freq_sorted])
plt.gca().set_xscale("log", nonposx='clip')
plt.gca().set_yscale("log", nonposy='clip')
plt.title('word distribution in Articles')
plt.gca().set_xlabel('rank')
plt.gca().set_ylabel('total appearances')
plt.savefig('Word_Distribution in Articles.jpg')



X = [[token for token in article.split()] for article in arts]
Y = [[token for token in headline.split()] for headline in heads]

plt.hist(map(len,Y),bins=50,rwidth=0.5);
axes = plt.gca()
axes.set_xlabel("Length of headline")
axes.set_ylabel("Number of headlines")
axes.set_title("Length of headline histogram")
axes.set_xlim([0,17])
plt.savefig('Length_Headline_Histogram.jpg')



plt.hist(map(len,X),bins=5000,rwidth=0.15);
axes = plt.gca()
axes.set_xlabel("Length of Article")
axes.set_ylabel("Number of Articles")
axes.set_title("Length of Article histogram")
axes.set_xlim([25,1000])
plt.savefig('Length_Article_Histogram.jpg')



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

