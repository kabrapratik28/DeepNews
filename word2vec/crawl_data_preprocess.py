#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import time
import codecs
import os
from preprocess import Preprocess_Text

class Preprocess_Crawl_Text(Preprocess_Text):
    def __init__(self,directory_name='../../data',temp_results='../../temp_results',raw_file_name='crawled_news_text.txt'):
        Preprocess_Text.__init__(self,directory_name,temp_results,raw_file_name)


    def parse_line_article(self, line):
        line = line.strip().split("#|#")
        if(len(line)<2 or len(line[0])==0 or len(line[1])==0):
        	return (None, None)        
        headline = line[0]
        text = line[1]
        return (headline,text)
        
    def generate_crawl_raw_file(self,is_separator=False):
        count = 0 
        malformed_articles= 0
        start_time = time.time()
        with codecs.open(self.raw_file_name, "w", encoding="utf-8") as f:
            for root, subdirs, files in os.walk(self.directory_name):
                for file_name in files:
                    #Mac file system file
                    if file_name==".DS_Store":
                        continue
                    print ("Working on file ",file_name)
                    file_path = os.path.join(root, file_name)
                    with codecs.open(file_path, encoding="utf-8") as each_file_pointer:
                        for each_line in each_file_pointer:
                            headline, text = self.parse_line_article(each_line)
                            if headline and text:
                                headline_tokens = self.tokenize(headline)
                                text_tokens = self.tokenize(text)
                                if is_separator:
                                    single_news_article = u" ".join(headline_tokens) + u"#|#" + u" ".join(text_tokens) + "\n"
                                else:
                                    single_news_article = u" ".join(headline_tokens) + u" ".join(text_tokens) + " " + self.eos_tag + "\n"
                                f.write(single_news_article)
                            else:
                                malformed_articles = malformed_articles + 1
                            count = count + 1
                            if count%10000==0:
                                print ("Processing done till ",count, "time took ",time.time()-start_time)
        print ("Total malformed articles ",malformed_articles)

def main():
    process_crawl_data = Preprocess_Crawl_Text()
    process_crawl_data.generate_crawl_raw_file()
    annotated_crawl_data = Preprocess_Crawl_Text(raw_file_name='annotated_crawled_news_text.txt')
    annotated_crawl_data.generate_crawl_raw_file(True)
    
if __name__ == "__main__":
    main()