import re
import argparse
import string
import math
from collections import defaultdict, Counter
from itertools import chain

# read file
def read_file(filename):
    with open (filename, 'r') as file:
        data = file.readlines()
    return data
    
def process_and_clean_data(data):
    query_ids = []
    query_content = []
    for i in range(len(data)):
        d = data[i]
        if d.startswith('.I ',0,3):
            query_ids.append(d[3:6])
        elif d == '.W\n':
            curr_query = []
            j = i+1
            while j<len(data) and not data[j].startswith('.I '):
                curr_query.extend(data[j].split())
                j += 1
            query_content.append(curr_query)
            
    closed_class_stop_words = ['a','the','an','and','or','but','about','above','after','along','amid','among',\
                           'as','at','by','for','from','in','into','like','minus','near','of','off','on',\
                           'onto','out','over','past','per','plus','since','till','to','under','until','up',\
                           'via','vs','with','that','can','cannot','could','may','might','must',\
                           'need','ought','shall','should','will','would','have','had','has','having','be',\
                           'is','am','are','was','were','being','been','get','gets','got','gotten',\
                           'getting','seem','seeming','seems','seemed',\
                           'enough', 'both', 'all', 'your' 'those', 'this', 'these', \
                           'their', 'the', 'that', 'some', 'our', 'no', 'neither', 'my',\
                           'its', 'his' 'her', 'every', 'either', 'each', 'any', 'another',\
                           'an', 'a', 'just', 'mere', 'such', 'merely' 'right', 'no', 'not',\
                           'only', 'sheer', 'even', 'especially', 'namely', 'as', 'more',\
                           'most', 'less' 'least', 'so', 'enough', 'too', 'pretty', 'quite',\
                           'rather', 'somewhat', 'sufficiently' 'same', 'different', 'such',\
                           'when', 'why', 'where', 'how', 'what', 'who', 'whom', 'which',\
                           'whether', 'why', 'whose', 'if', 'anybody', 'anyone', 'anyplace', \
                           'anything', 'anytime' 'anywhere', 'everybody', 'everyday',\
                           'everyone', 'everyplace', 'everything' 'everywhere', 'whatever',\
                           'whenever', 'whereever', 'whichever', 'whoever', 'whomever' 'he',\
                           'him', 'his', 'her', 'she', 'it', 'they', 'them', 'its', 'their','theirs',\
                           'you','your','yours','me','my','mine','I','we','us','much','and/or'
                           ]
                           
    non_stop_word_count = 0
    new_query_content = []
    for sentence in query_content:
        new_sentence = []
        for word in sentence:
            if word in closed_class_stop_words or word in string.punctuation or word.isnumeric() or word.isdigit():
                sentence.remove(word)
            else:
                new_word = word.lstrip(',)./')
                new_sentence.append(new_word)
                non_stop_word_count += 1
                #calculate IDF
        new_query_content.append(new_sentence)
    
    return new_query_content, non_stop_word_count
    
def compute_idf(text):
    num_docs = len(text)
    doc_freq = defaultdict(int)
    for doc in text:
        unique_terms = set(doc)
        for term in unique_terms:
            doc_freq[term] += 1
            
    idf = []
    for doc in text:
        curr_dict = {}
        for term in doc:
            count = doc_freq[term]
            if count>0:
                curr_dict[term] = math.log(num_docs/float(count))
            else:
                curr_dict[term] = 0
        idf.append(curr_dict)
    return idf
        
def compute_td_idf(text, idf):
    td_idf = []

    index = 0
    for sentence in idf:
        tf = Counter(text[index])
        curr_dict = {}
        for term, count in sentence.items():
            curr_dict[term] = count * (tf[term]/len(sentence))
        td_idf.append(curr_dict)
        index += 1

    return td_idf


def cosine_similarity(vec1, vec2):
    common_terms = set(vec1.keys()).intersection(set(vec2.keys()))
    
    vector = []
    
    dot_product = 0
    for term in common_terms:
        dot_product += (vec1[term] * vec2[term])
    
    norm_vec1 = 0
    for val in vec1.values():
        norm_vec1 += val**2
    
    norm_vec2 = 0
    for val in vec2.values():
        norm_vec2 += val**2
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    
    return dot_product/(math.sqrt(norm_vec1*norm_vec2))
    
            

def main():
    # process query
    data_query = read_file('cran.qry')
    query_content, non_stop_word_count_query = process_and_clean_data(data_query)
    idf_query = compute_idf(query_content)
    td_idf_query = compute_td_idf(query_content, idf_query)
        
    # process abstract
    data_abstract = read_file('cran.all.1400')
    abstract_content, non_stop_word_count_abstract = process_and_clean_data(data_abstract)
    idf_abstract = compute_idf(abstract_content)
    td_idf_abstract = compute_td_idf(abstract_content, idf_abstract)
    
    # calculate cosine similarity
    cosines = {}
    index_query = 1
    for each_query in td_idf_query:
        curr_query = {}
        index_ab = 1
        for each_abstract in td_idf_abstract:
            curr_cosine = cosine_similarity(each_query, each_abstract)
            curr_query[index_ab] = curr_cosine
            index_ab += 1
        cosines[index_query] = curr_query
        index_query += 1

    # rank & print
    file = open('output.txt', 'w')
            
    for query_id, each_cosine in cosines.items():
        sorted_cosines = dict(sorted(each_cosine.items(), key=lambda item: item[1], reverse=True))
        
        for abstract_id, each_abstract in sorted_cosines.items():
            if each_abstract == 0.0:
                file.write(str(query_id) + " " + str(abstract_id) + " 0\n")
            else:
                file.write(str(query_id) + " " + str(abstract_id) + " " + str(each_abstract)+ "\n")
            
    



if __name__ == '__main__':
    main()
