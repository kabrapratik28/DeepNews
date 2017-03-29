#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import time
import codecs
import os
from preprocess import Preprocess_Text

class Preprocess_Crawl_Text(Preprocess_Text):
    def __init__(self,directory_name='../../Data',temp_results='../../temp_results',raw_file_name='crawled_news_text.txt'):
        Preprocess_Text.__init__(self,directory_name,temp_results,raw_file_name)
        
    def generate_crawl_raw_file(self,):
        count = 0 
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
                            headline_tokens = self.tokenize(each_line)
                            single_news = u" ".join(headline_tokens) + u"\n"
                            f.write(single_news)
                            count = count + 1
                            if count%10000==0:
                                print ("Processing done till ",count, "time took ",time.time()-start_time)

def main():
    process_crawl_data = Preprocess_Crawl_Text()
    process_crawl_data.generate_crawl_raw_file()
    
if __name__ == "__main__":
    main()