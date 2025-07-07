#!/usr/bin/env python
import sys

from lexsub_xml import read_lexsub_xml
from lexsub_xml import Context 

# suggested imports 
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords

import numpy as np
import tensorflow as tf

import gensim
import transformers 

from typing import List
###added import strign lol
import string 
#tensorflow.compat.v1.disable_eager_execution()
#from collections import defaultdict
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
## UNI: yd2696

def tokenize(s): 
    """
    a naive tokenizer that splits on punctuation and whitespaces.  
    """
    s = "".join(" " if x in string.punctuation else x for x in s.lower())    
    return s.split() 

def get_candidates(lemma, pos) -> List[str]:
    # Part 1
    #string in line 23 bugging me lol
    # look at wn.lemmas bit from 
    #l1 = wn.lemmas('break', pos='n')
    #s1 = l1.synset()
    ret = []
    dup = set()
    dup.add(lemma)
    lem = wn.lemmas(lemma,pos)
    for i in range(0,len(lem)):
        syn = lem[i].synset()
        syn_name = syn.name()
        new_lemma = syn.lemmas()
        for j in range(0,len(new_lemma)):
            cur = new_lemma[j].name()
            if cur in dup:
                pass
            else:
                if "_" in cur:
                    cur = cur.replace("_"," ")
                #elif "-" in cur:
                #    cur = cur.replace("-"," ")
                #print(new_lemma)
                dup.add(cur)
                ret.append(cur)
    ## find synset() for each in lem
    #for debugging 
    #print(dup)
    return ret

def smurf_predictor(context : Context) -> str:
    """
    suggest 'smurf' as a substitute for all words.
    """
    return 'smurf'

def wn_frequency_predictor(context : Context) -> str:
    # find synset for target word
    # iterate through each all synset
    # find lemma of each and take find count
    ret = ""
    valid = get_candidates(context.lemma,context.pos)
    lexeme = dict()
    for i in range(0,len(valid)):
        lexeme[valid[i]] = 0
    dup = set()
    dup.add(context.lemma)

    lem = wn.lemmas(context.lemma,context.pos)
    for i in range(0,len(lem)):
        syns = lem[i].synset()
        syn_name = syns.name()
        new_lemma = syns.lemmas()
        for j in range(0,len(new_lemma)):
            cur = new_lemma[j].name()
            if "_" in cur:
                    cur = cur.replace("_"," ")
            if cur in dup and context.lemma != cur:
                lexeme[cur] += new_lemma[j].count()
            elif cur not in dup:
                dup.add(cur)
                lexeme[cur] = new_lemma[j].count()
                

    lexeme_keys = np.array(list(lexeme.keys()))
    lexeme_values = np.array(list(lexeme.values())) 
    ### key:value -> name:count
    max_index =  np.argmax(lexeme_values)
    ret = lexeme_keys[max_index]
    return ret #None # replace for part 2

def wn_simple_lesk_predictor(context : Context) -> str:
    # apparently can use get_canidates() here
    # to help remove stop_words 
    stop_words = set(stopwords.words('english')) 
    # for each context object compare both the left and right context to definitions of each in the synset
    # synset with highest matching words provide most similar word/ are the -> ret
    ret = ''
    res = {}
    left_c = context.left_context
    right_c = context.right_context
    full_c = left_c + right_c
    full_c = " ".join(full_c)
    context_tok = tokenize(full_c)
    full_context = list()
    for w in context_tok:
        if w not in stop_words:
            full_context.append(w)
    ## find syn with most matches 
    ## class hypernyms
        ## -> definition() and examples
        # current context 
        # first set current context 
        # second combine into one string 
        ## length of overlap set is score
    max_score = 0
    max_lexeme = None   
    lem = wn.lemmas(context.lemma,context.pos)
    for i in range(0,len(lem)):
        syn = lem[i].synset()
        syn_name = syn.name()
        phrase = syn.definition()
        new_lemma = syn.lemmas()
        #syn_toks = tokenize(syn.definiton())
        syn_context = list()
        count = 0
        ex = syn.examples() ## list of strings 
        hyp = syn.hypernyms() ## list
        ### example loop ###
        for j in range(0,len(ex)):
            phrase = phrase + " "+ ex[j]
        ### hypernym loop ###
        for k in range(0,len(hyp)):
            curr = hyp[k] 
            phrase = phrase + " "+curr.definition()
            curr_ex = curr.examples()
            for l in range(0,len(curr_ex)):
                phrase = phrase + " "+curr_ex[l]
        ####################################
        # remove stop words and compute overlap
        syn_toks = tokenize(phrase)
        for w in syn_toks:
            if w not in stop_words:
                syn_context.append(w)
        overlap = set(syn_context).intersection(full_context)
        #res[syn_name] = len(overlap)
        ## every synset look at each lemma()
        #check if lemma is my target?
        # keep track of this count if true ->B
        #print(syn_name+'.'+context.lemma)
        B = 0#wn.lemma(syn_name+'.'+context.lemma).count()
        C = 0
        #tie ={}
        
        for s in range(0,len(new_lemma)):
            name = new_lemma[s].name()
            if i== 0:
                B = new_lemma[s].count()
            if name == context.lemma:
                pass
                ## update B
                #if B < new_lemma[s].count():
                B = new_lemma[s].count()
            else:
                ## update C
                #if C< new_lemma[s].count():
                C = new_lemma[s].count()
                score = 1000*len(overlap) + 100*B + C
                if score >max_score:
                    max_score = score
                    max_lexeme = name
    ret = max_lexeme
    return ret #replace for part 3        

class Word2VecSubst(object):
        
    def __init__(self, filename):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(filename, binary=True)    

    def predict_nearest(self,context : Context) -> str:
        ret = ''
        syn = get_candidates(context.lemma,context.pos)
        #target = self.model.get_vector(context.lemma)
        #print(syn)
        vals = dict()
        for i in range(0,len(syn)):
            #curr = self.model.get_vector(syn[i])
            #print(target)
            #print("##############")
            #print(curr)
            if syn[i] in self.model.key_to_index: 
                similarity = self.model.similarity(context.lemma,syn[i])#self.model.similarity(target,curr)
                vals[syn[i]] = similarity
        #print(vals)
        keys = np.array(list(vals.keys()))
        values = np.array(list(vals.values()))
        max_index = np.argmax(values)
        #print(keys)
        #print(values)
        #print("######")
        #print(max_index)
        ret = keys[max_index]
        return ret # replace for part 4


class BertPredictor(object):

    def __init__(self): 
        self.tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = transformers.TFDistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')
        ############
        # part 6 model belos
        gpt_style = 'gpt2'
        self.gpt_tokenizer = transformers.AutoTokenizer.from_pretrained(gpt_style)
        self.gpt_model = transformers.TFGPT2LMHeadModel.from_pretrained(gpt_style, from_pt=True)
        # wont work if pytorch is not installed 
        #transformers.AutoModelForCausalLM.from_pretrained(gpt_style) -> needs pytorch
        '''
        tokenizer = transformers.GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B')
        # Load the GPT-Neo model
        model = transformers.TFGPT2LMHeadModel.from_pretrained('EleutherAI/gpt-neo-1.3B')
        '''
    def predict(self, context : Context) -> str:
        canidates = get_candidates(context.lemma,context.pos)
        ret = ""
        left_c = " ".join(context.left_context)
        right_c = ' '.join(context.right_context)
        
        full_c = left_c + ' [MASK] ' + right_c
        
        mask_index = len(self.tokenizer.tokenize(left_c)) + 1
        
        input_toks = self.tokenizer.encode(full_c)
        input_mat = np.array(input_toks).reshape((1,-1))  # get a 1 x len(input_toks) matrix
        outputs = self.model.predict(input_mat,verbose = None)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0,mask_index])[::-1] # Sort in increasing order
        #print(best_words)
        
        res = self.tokenizer.convert_ids_to_tokens(best_words)
        #print(res)
        for i in range(0,len(res)):
            if res[i] in canidates:
                #scores[res[i]] = best_words[i]   
                ret = res[i]
                break 
        return ret # replace for part 5
    
    def part6(self, context : Context) -> str:
        ### Essentially the same as part5
        ## but with a gpt2 model instead of bert 
        ## performs worse but seemed interesting to do with a different model
        canidates = get_candidates(context.lemma,context.pos)
        ret = ""
        left_c = " ".join(context.left_context)
        right_c = ' '.join(context.right_context)
        
        full_c = left_c + ' [MASK] ' + right_c
        
        mask_index = len(self.gpt_tokenizer.tokenize(left_c)) +1 
        
        input_toks = self.gpt_tokenizer.encode(full_c)
        input_mat = np.array(input_toks).reshape((1,-1))  # get a 1 x len(input_toks) matrix
        outputs = self.gpt_model.predict(input_mat,verbose = None)
        predictions = outputs[0]
        best_words = np.argsort(predictions[0,mask_index])[::-1] # Sort in increasing order
        #print(best_words)
        
        res = self.tokenizer.convert_ids_to_tokens(best_words)
        #print(res)
        for i in range(0,len(res)):
            if res[i] in canidates:
                #scores[res[i]] = best_words[i]   
                ret = res[i]
                break 
        return ret # replace for part 5
    

if __name__=="__main__":

    # At submission time, this program should run your best predictor (part 6). 
    W2VMODEL_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'## needed for part 4
    
    predictor = Word2VecSubst(W2VMODEL_FILENAME) # part 4 needed
    pred_part5 = BertPredictor() # part5/part6 needed
    #####################################
    #####################################
    #python lexsub_main.py lexsub_trial.xml  > smurf.predict
    #gc smurf.predict -head 10
    #### My part 4 and 5 perform best and have idenitcal results
    # to test part5/part6 uncomment  
    for context in read_lexsub_xml(sys.argv[1]):
        #print(context)  # useful for debugging
        #prediction = smurf_predictor(context)          #p1
        #prediction = wn_frequency_predictor(context)   #p2
        #prediction = wn_simple_lesk_predictor(context) #p3
        #prediction = predictor.predict_nearest(context) #p4
        prediction = pred_part5.predict(context)       #p5
        #prediction = pred_part5.part6(context)         #p6
        print("{}.{} {} :: {}".format(context.lemma, context.pos, context.cid, prediction))
