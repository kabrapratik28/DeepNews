#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os
import re
import codecs
import time
import xml.etree.ElementTree as ET

class Preprocess_Text(object):
    def __init__(self,directory_name='../../data',temp_results='../../temp_results',raw_file_name='raw_news_text.txt'):
        self.directory_name = directory_name
        self.temp_results = temp_results
        if not os.path.exists(temp_results):
            os.makedirs(temp_results)
        self.raw_file_name = os.path.join(temp_results,raw_file_name)
        self.eos_tag = '<eos>'        
    
    def is_this_string(self,s):
        if isinstance(s, str):
            return True
        elif isinstance(s, unicode):
            return True
        else:
            return False
    
    def parse_xml(self,file_name):
        headline = ""
        text = ""
        try:
            root = ET.parse(file_name)
            nodeHL = root.find('HL')
            nodeText = root.find('TEXT')
            if nodeHL!=None:    
                headline = nodeHL.text
            if nodeText!=None:  
                text = nodeText.text
        except:
            print("Error in file ",file_name)
        return (headline, text)
        
    def tokenize(self,data_string):
        if(not self.is_this_string(data_string)):
            data_string = ""
        tokens=re.split(u'[ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ`\-=~!@#$%^&*()_+\[\]{};\'\\:"|<,./<>?‘’“”]|\n|।', data_string)
        tokens=filter(None,tokens)
        return tokens

    def generate_raw_file(self,is_separator=False):
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
                    if is_separator:
                        single_news = u" ".join(headline_tokens) + u"#|#" + u" ".join(text_tokens) + "\n"
                    else:
                        single_news = u" ".join(headline_tokens) + u" ".join(text_tokens) + " " + self.eos_tag + "\n"
                    f.write(single_news)
                    count = count + 1
                    if count%10000==0:
                        print ("Processing done till ",count, "time took ",time.time()-start_time)

def main():
    process_data = Preprocess_Text()
    process_data.generate_raw_file()
    annotate_data = Preprocess_Text(raw_file_name='annotated_news_text.txt')
    annotate_data.generate_raw_file(True)

    
if __name__ == "__main__":
    main()