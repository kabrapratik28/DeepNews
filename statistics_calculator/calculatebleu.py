# -*- coding: utf-8 -*-
import sys
import codecs
from math import exp, log
from os import listdir
from os.path import isfile, join, isdir
from collections import Counter
no_of_grams=4

def extract_all_files_in_dir(dir_name):
    onlyfiles = [join(dir_name, f) for f in listdir(dir_name) if f!=".DS_Store" and isfile(join(dir_name, f))]
    return onlyfiles

def file_line_counter(file_name):
    with codecs.open(file_name, 'r',encoding='utf-8') as f:
        for i, l in enumerate(f):
            pass
    return i+1

def file_opener(candidate_file, reference_files):
    candidate_file_fp = codecs.open(candidate_file,'r',encoding='utf-8')
    reference_files_fp = []
    for i in reference_files:
        ref_file_fp = codecs.open(i,'r',encoding='utf-8')
        reference_files_fp.append(ref_file_fp)
    return candidate_file_fp, reference_files_fp

def file_closer(candidate_file_fp, reference_files_fp):
    for i in [candidate_file_fp,]+reference_files_fp:
        try:
            i.close()
        except:
            None

def process_sentence(candidate_sentence, reference_sentences):
    candidate_sentence_list = candidate_sentence.strip().split()
    reference_sentences_list = [x.strip().split() for x in reference_sentences]
    return candidate_sentence_list, reference_sentences_list
            
def n_gram_generator(word_list, n_gram):
    length_of_words = len(word_list)
    tokens = []
    for i in range(length_of_words-n_gram+1):
        token = " ".join(word_list[i:i+n_gram])
        tokens.append(token)
    return tokens    

def modified_precision(candidate_sentence_list, reference_sentences_list, n_gram):
    candidate_tokens = n_gram_generator(candidate_sentence_list, n_gram)
    reference_sentences_tokens = [n_gram_generator(x,n_gram) for x in reference_sentences_list]
    candidate_tokens_count = Counter(candidate_tokens)
    reference_sentences_tokens_count = [Counter(x) for x in reference_sentences_tokens]
    sum_of_tokens_appeared = 0
    for token, count in candidate_tokens_count.iteritems():
        max_count_occured_ref = 0
        for each_ref in reference_sentences_tokens_count:
            if token in each_ref:
                max_count_occured_ref = max(max_count_occured_ref,each_ref[token])
        clipped_count = min(max_count_occured_ref, count)
        sum_of_tokens_appeared += clipped_count
    return float(sum_of_tokens_appeared), float(len(candidate_tokens))

def brevity_raw(candidate_sentence_list, reference_sentences_list):
    mindiff = sys.maxsize
    can_len = len(candidate_sentence_list)
    res_len = 0
    for i in reference_sentences_list:
        len_i = len(i)
        c_diff = abs(len_i-can_len)
        if (c_diff < mindiff) or (c_diff == mindiff and len_i < res_len):
            res_len = len_i
            mindiff = c_diff
    return float(can_len), float(res_len)

def brevity(corpus_len, reference_len):
    if corpus_len > reference_len:
        return 1.0
    else:
        return exp(1-(reference_len/corpus_len))

def bleu_calcuator(candidate_file, reference_files):
    no_of_lines = file_line_counter(candidate_file)
    candidate_file_fp, reference_files_fp = file_opener(candidate_file, reference_files)
    n_gram_precision = {}
    candidate_len, reference_len = 0.0 , 0.0 
    for i in range(1,no_of_grams+1):
        n_gram_precision[i] = {"num":0.0,"den":0.0}
    for i in range(no_of_lines):
        c_sentence = candidate_file_fp.readline().strip()
        r_sentences = [x.readline().strip() for x in reference_files_fp]
        candidate_sentence_list, reference_sentences_list = process_sentence(c_sentence, r_sentences)
        can_len, res_len = brevity_raw(candidate_sentence_list, reference_sentences_list)
        candidate_len += can_len
        reference_len += res_len
        for j in range(1,no_of_grams+1):
            numerator, denometer = modified_precision(candidate_sentence_list,
                                                      reference_sentences_list,n_gram=j)
            n_gram_precision[j]["num"] += numerator
            n_gram_precision[j]["den"] += denometer
        
    file_closer(candidate_file_fp, reference_files_fp)
    
    bp = brevity(candidate_len, reference_len)
    total_pn = 0.0
    for i in range(1,no_of_grams+1):
        total_pn += ((1/float(no_of_grams))*log(n_gram_precision[i]["num"]/n_gram_precision[i]["den"]))
    bleu_score = bp * exp(total_pn)
    with open("bleu_out.txt","w") as fp:
        fp.write(str(bleu_score))

def main():
    file_name = sys.argv[1]
    cand_file = "cand.txt"
    ref_file = "ref.txt"
    cf = codecs.open(cand_file,"w",encoding='utf-8')
    rf = codecs.open(ref_file,"w",encoding='utf-8')
    with codecs.open(file_name,encoding='utf-8') as fp:
		for each_line in fp:
			data = each_line.strip().split("#|#")
			rf.write(data[0]+"\n")
			cf.write(data[1]+"\n")
    cf.close()
    rf.close()
    candidate_file = cand_file
    reference_file = ref_file

    reference_files = []
    #check is directory or file given ...
    if isdir(reference_file):
        reference_files = extract_all_files_in_dir(reference_file)
    else:
        reference_files = [reference_file]
    bleu_calcuator(candidate_file, reference_files)
    
if __name__ == '__main__':
    main()