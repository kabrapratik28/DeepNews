#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import re
import codecs
import time
import xml.etree.ElementTree as ET

class Preprocess_Text(object):
    def __init__(self,directory_name='../../Data',temp_results='../../temp_results',raw_file_name='raw_news_text.txt'):
        self.directory_name = directory_name
        self.temp_results = temp_results
        if not os.path.exists(temp_results):
            os.makedirs(temp_results)
        self.raw_file_name = os.path.join(temp_results,raw_file_name)        
        
    def parse_xml(self,file_name):
        headline = ""
        text = ""
        root = ET.parse(file_name)
        nodeHL = root.find('HL')
        nodeText = root.find('TEXT')
        if nodeHL!=None:    
            headline = nodeHL.text
        if nodeText!=None:  
            text = nodeText.text
        return (headline, text)
        
    def tokenize(self,data_string):
        text = data_string
        tokens=re.split(u'[ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?]|\n|।', text)
        tokens=filter(None,tokens)
        return tokens

    def generate_raw_file(self,):
        count = 0 
        start_time = time.time()
        with codecs.open(self.raw_file_name, "w", encoding="utf-8") as f:
            for root, subdirs, files in os.walk(self.directory_name):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    #Mac file system file
                    if file_name==".DS_Store":
                        continue
                    headline, text = self.parse_xml(file_path)
                    headline_tokens =  self.tokenize(headline)
                    text_tokens = self.tokenize(text)                    
                    single_news = u" ".join(headline_tokens) + u" ".join(text_tokens) + "\n"
                    f.write(single_news)
                    count = count + 1
                    if count%100000==0:
                        print ("Processing done till ",count, "time took ",time.time()-start_time)

def main():
    process_data = Preprocess_Text()
    process_data.generate_raw_file()
    
if __name__ == "__main__":
    main()