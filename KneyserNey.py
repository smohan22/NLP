import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize
from math import log10

class kneyserNey():
    
    '''import pandas as pd
    import numpy as np
    import nltk
    from math import log10
    '''
    def __init__(self):
        #setting in fit
        self.ngram_order = None
        self.all_gram = None
        self.vocab = None
    
    
    def make_ngrams(self, text, n):
        '''
        takes a text and n gram and creates an n-gram for it
        '''
        list_ngram = []
        # Parse text into sentences
        sent_text = sent_tokenize(text)
        # Get n-grams
        for sentence in sent_text:
            sentence = (n-1)*"<s> " + sentence # create n-1 pseudo tokens
            n_grams = nltk.ngrams(sentence.split(), n)
            for grams in n_grams:
                new_gram = []
                for word in grams:
                    word = word.strip(".").strip("?").strip("!").strip(";").strip(":").strip('"')
                    wLower = word.lower()
                    new_gram.append(wLower)
                list_ngram.append(tuple(new_gram))
        return list_ngram

    
    def make_ngrams_counts(self, list_ngrams, n):
        '''
        takes a text and n gram and creates an n-gram for it
        '''
        dict_ngram = {}

        for ngram in list_ngrams:
            if ngram in dict_ngram:
                dict_ngram[ngram] = dict_ngram[ngram] + 1 
            else:
                dict_ngram[ngram] = 1
        return dict_ngram

    
    def fit(self, text, ngram_order):
        '''
        Create the fit database based on the order
        '''
        all_gram = {}
        for n in range(1, ngram_order+1):
            list_ngrams = self.make_ngrams(text, n)
            all_gram[n] = self.make_ngrams_counts(list_ngrams, n)
        vocab = len(all_gram[1]) -1 # -1 to take care of start phrase
        self.ngram_order = ngram_order
        self.all_gram = all_gram
        self.vocab = vocab
        return self
    
    def score(self, text, n, d):
        '''
        Performs basic checks before proceeding to calculate score of the phrase
        '''
        if (d <= 0 ) or (d >= 1):
            return "Please discounting a value between 0 and 1"
        else:
            list_ngrams = self.make_ngrams(text, n) #makes ngram tuples
            log_prob = 0
            for phrase in list_ngrams:
                log_prob += log10(self.calculate_score(phrase, d))
            return log_prob
            
        
    def calculate_score(self, phrase, d):
        '''
        Calculate the calculate_score based on the phrase of reference
        '''
        ngram_len = len(phrase)
        all_gram = self.all_gram
        ngram_order = self.ngram_order
        vocab = self.vocab
        if ngram_len == 1: # base case
            if phrase in all_gram[ngram_len]:
                probability = all_gram[ngram_len][phrase]/vocab
                return probability
            else:
                return 1/(vocab + 1) # the word does not exist
        else: #recursive case    
            if ngram_len == ngram_order: ##counting  case
                if phrase in all_gram[ngram_len]:
                    num_1 = max(all_gram[ngram_len][phrase] - d, 0)
                    num_2 = len([each for each in all_gram[ngram_len] if phrase[:-1] == each[:-1]])
                    denom = sum([all_gram[ngram_len][each] for each in all_gram[ngram_len] if phrase[:-1] == each[:-1]])
                    probability = num_1/denom + d*num_2/denom*self.calculate_score(phrase[1:], d)
                    return probability
                else:
                    probability = self.calculate_score(phrase[1:], d) # we check for one lower gram
                    return probability

            else: #continuous counting case
                if phrase in all_gram[ngram_len]:
                    num_1 = max(len([each for each in all_gram[ngram_len+1] if phrase == each[1:]]) - d, 0)
                    num_2 = len([each for each in all_gram[ngram_len] if phrase[:-1] == each[:-1]])
                    denom = len([each for each in all_gram[ngram_len + 1 ] if phrase[:-1] == each[1:-1]])
                    probability = num_1/denom + d*num_2/denom*self.calculate_score(phrase[1:], d)
                    return probability
                else:
                    probability = self.calculate_score(phrase[1:], d) # we check for one lower gram
                    return probability