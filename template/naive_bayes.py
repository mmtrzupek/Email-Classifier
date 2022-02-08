# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import numpy as np
import math
from tqdm import tqdm
from collections import Counter
import reader

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""




"""
  load_data calls the provided utility to load in the dataset.
  You can modify the default values for stemming and lowercase, to improve performance when
       we haven't passed in specific values for these parameters.
"""
 
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming is {stemming}")
    print(f"Lowercase is {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset_main(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


def create_word_maps_uni(X, y, max_size=None):
    """
    X: train sets (a list of emails; each email is a list of words)
    y: train labels (a list of labels, one label per email; each label is 1 or 0)
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: words 
        values: number of times the word appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word appears 
    """
    print("LENGTH OF THE TRAIN SETS")
    print(len(X),'X')

    pos_vocab = {}
    neg_vocab = {}

    # X: train set = [['random', 'word', 'example'], ['another', 'word'], []]
    # y: train label := [1, 0]
    
    # Iterate over the list of emails
    for i in range(len(X)):
        # Iterate over list of each individual word in the email
        for j in X[i]:
            # if the train label at index is positive, count number of times word appears
            if y[i] == 1:
                pos_vocab[j] = pos_vocab.get(j,0) + 1
            # if the train label at index is negative
            else:
                neg_vocab[j] = neg_vocab.get(j,0) + 1

    return dict(pos_vocab), dict(neg_vocab)


def create_word_maps_bi(X, y, max_size=None):
    """
    X: train sets (a list of emails; each email is a list of words)
    y: train labels (a list of labels, one label per email; each label is 1 or 0)
    max_size: you can ignore this, we are not using it

    return two dictionaries: pos_vocab, neg_vocab
    pos_vocab:
        In data where labels are 1 
        keys: pairs of words
        values: number of times the word pair appears
    neg_vocab:
        In data where labels are 0
        keys: words 
        values: number of times the word pair appears 
    """
    #print(len(X),'X')
    pos_vocab = {}
    neg_vocab = {}

    # X: train set = [['random', 'word', 'example'], ['another', 'word'], []]
    # y: train label := [1, 0]
    # {'random word' : 1, 'word example' : 0}

    for i in range(len(X)):
        for j in range(len(X[i]) - 1):
            bi_word = X[i][j] + " " + X[i][j+1]
            if y[i] == 1:
                pos_vocab[bi_word] = pos_vocab.get(bi_word, 0) + 1
            else:
                neg_vocab[bi_word] = neg_vocab.get(bi_word, 0) + 1

    uni_map_pos, uni_map_neg = create_word_maps_uni(X, y)

    return Merge(dict(pos_vocab), dict(uni_map_pos)), Merge(dict(neg_vocab), dict(uni_map_neg))


# Helper Function - Return two merged dictionaries
# Referenced from https://www.geeksforgeeks.org/python-merging-two-dictionaries/
def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res

# Keep this in the provided template
def print_paramter_vals(laplace,pos_prior):
    print(f"Unigram Laplace {laplace}")
    print(f"Positive prior {pos_prior}")


"""
You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
Notice that we may pass in specific values for these parameters during our testing.
"""

def naiveBayes(train_set, train_labels, dev_set, laplace=0.001, pos_prior=0.8, silently=False):
    '''
    Compute a naive Bayes unigram model from a training set; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    # Keep this in the provided template
    print_paramter_vals(laplace,pos_prior)

    # train set = [['random', 'word', 'example'], ['another', 'word'], []]
    # train label := [1, 0]

    # P(Y = ham) is 1 - pos_proir
    # P(Y = spam) = pos_prior

    result = []
    # spam = 1 (pos_vocab), ham = 0 (neg_vocab) 

    spam, ham = create_word_maps_uni(train_set, train_labels, max_size=None)

    # Total number of words in Ham/Spam training set
    total_num_words_spam = sum(spam.values())
    total_num_words_ham = sum(ham.values())

    spam_probability = {} # prob of word being spam P(X=x | Y=Spam)
    ham_probability = {} # prob of word being ham P(X=x | Y=Ham)

    # -------------------- TRAINING PHASE --------------------
    # Laplace smoothing
    for word in spam:
        spam_probability[word] = (spam[word] + laplace) / (total_num_words_spam + laplace*(1 + len(spam)))

    for word in ham:
        ham_probability[word]  = (ham[word] + laplace) / (total_num_words_ham + laplace*(1 + len(ham)))

    # -------------------- DEVELOPMENT PHASE --------------------
    for email in dev_set:
        # Prevent underflow by taking the logs
        ham_updated_prior = np.log(1-pos_prior) # prior prob that email is ham
        spam_updated_prior = np.log(pos_prior) # prior prob that email is spam

        for word in email:
            if word in ham_probability:
                ham_updated_prior += np.log(ham_probability[word])
            # If we see an out-of-vocab (OOV) word
            else:
                ham_updated_prior += np.log((laplace) / (total_num_words_ham + laplace*(1 + len(ham))))

            if word in spam_probability:
                spam_updated_prior += np.log(spam_probability[word])
            # If we see an out-of-vocab (OOV) word
            else:
                spam_updated_prior += np.log((laplace) / (total_num_words_spam + laplace*(1 + len(spam))))

        if ham_updated_prior > spam_updated_prior:
            result.append(0)
        else:
            result.append(1)

    return result


# Keep this in the provided template
def print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior):
    print(f"Unigram Laplace {unigram_laplace}")
    print(f"Bigram Laplace {bigram_laplace}")
    print(f"Bigram Lambda {bigram_lambda}")
    print(f"Positive prior {pos_prior}")


def bigramBayes(train_set, train_labels, dev_set, unigram_laplace=0.001, bigram_laplace=0.005, bigram_lambda=0.5,pos_prior=0.8,silently=False):
    '''
    Compute a unigram+bigram naive Bayes model; use it to estimate labels on a dev set.

    Inputs:
    train_set = a list of emails; each email is a list of words
    train_labels = a list of labels, one label per email; each label is 1 or 0
    dev_set = a list of emails
    unigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating unigram probs
    bigram_laplace (scalar float) = the Laplace smoothing parameter to use in estimating bigram probs
    bigram_lambda (scalar float) = interpolation weight for the bigram model
    pos_prior (scalar float) = the prior probability of the label==1 class
    silently (binary) = if True, don't print anything during computations 

    Outputs:
    dev_labels = the most probable labels (1 or 0) for every email in the dev set
    '''
    print_paramter_vals_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    max_vocab_size = None
    result = []
    spam_uni, ham_uni = create_word_maps_uni(train_set, train_labels, max_size=None)
    spam_bi, ham_bi = create_word_maps_bi(train_set, train_labels, max_size=None)

    # Count # of occurrences of word in each dict
    total_num_words_spam_unigram = sum(spam_uni.values())
    total_num_words_ham_unigram = sum(ham_uni.values())
    total_num_words_spam_bigram = sum(spam_bi.values())
    total_num_word_ham_bigram = sum(ham_bi.values())

    # Dictionaries that hold words and their probablilites of being spam/ham
    spam_prob_uni = {}
    ham_prob_uni = {}
    spam_prob_bi = {}
    ham_prob_bi = {}

    for word in spam_uni:
        spam_prob_uni[word] = np.log((spam_uni[word] + unigram_laplace)/(total_num_words_spam_unigram + unigram_laplace*(1+len(spam_uni))))
    for word in ham_uni:
        ham_prob_uni[word] = np.log((ham_uni[word] + unigram_laplace)/(total_num_words_ham_unigram + unigram_laplace*(1+len(ham_uni))))
    for word in spam_bi:
        spam_prob_bi[word] = np.log((spam_bi[word] + bigram_laplace)/(total_num_words_spam_bigram + bigram_laplace*(1+len(spam_bi))))
    for word in ham_bi:
        ham_prob_bi[word] = np.log((ham_bi[word] + bigram_laplace)/(total_num_word_ham_bigram + bigram_laplace*(1+len(ham_bi))))

    spam_prior = np.log(pos_prior)
    ham_prior = np.log(1 - pos_prior)

    # Lists which will contain the probabilities of Uni/Bi Spam/Ham emails
    spam_uni_list = []
    ham_uni_list = []
    spam_bi_list = []
    ham_bi_list = []
    
    for email in range(len(dev_set)):
        emails = dev_set[email]
        uni_spam_prior = spam_prior
        uni_ham_prior = ham_prior
        # Unigrams
        for word in emails:
            # Spam Unigram
            if word in spam_prob_uni:
                uni_spam_prior += spam_prob_uni[word]
            # OOV word
            else:
                uni_spam_prior += np.log((unigram_laplace)/(total_num_words_spam_unigram + unigram_laplace*(1+len(spam_uni))))

            # Ham Unigram
            if word in ham_prob_uni:
                uni_ham_prior += ham_prob_uni[word]
            # OOV word
            else:
                uni_ham_prior += np.log((unigram_laplace)/(total_num_words_ham_unigram + unigram_laplace*(1+len(ham_uni))))

        # Add Spam/Ham Unigram probabilities to lists
        spam_uni_list.append(uni_spam_prior)
        ham_uni_list.append(uni_ham_prior)

        # Bigrams
        bi_spam_prior = spam_prior
        bi_ham_prior = ham_prior

        for i in range(len(emails) - 1):
            bigram_words = emails[i] + ' ' + emails[i + 1]
            # Spam Bigram
            if bigram_words in spam_prob_bi:
                bi_spam_prior += spam_prob_bi[bigram_words]
            # OOV word
            else:
                bi_spam_prior += np.log((bigram_laplace)/(total_num_words_spam_bigram + bigram_laplace*(1+len(spam_bi))))
            # Ham Bigram
            if bigram_words in ham_prob_bi:
                bi_ham_prior += ham_prob_bi[bigram_words]
            # OOV word 
            else:
                bi_ham_prior += np.log((bigram_laplace)/(total_num_word_ham_bigram + bigram_laplace*(1+len(ham_bi))))

        spam_bi_list.append(bi_spam_prior)
        ham_bi_list.append(bi_ham_prior)

        # lhs is prob of email being spam from mixture model
        lhs = (bigram_lambda*spam_bi_list[email]) + ((1 - bigram_lambda)*spam_uni_list[email])
        # rhs is prob of email being ham from mixture model
        rhs = (bigram_lambda*ham_bi_list[email]) + ((1 - bigram_lambda)*ham_uni_list[email])

        if lhs >= rhs:
            result.append(1)
        else:
            result.append(0)

    return result