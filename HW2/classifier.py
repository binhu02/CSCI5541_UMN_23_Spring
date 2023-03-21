# Author: Bin Hu
# HW2 for CSCI 5541 23S

import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import wordpunct_tokenize as word_tokenize
from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import pad_both_ends, padded_everygram_pipeline
from nltk.util import trigrams
from collections import defaultdict
import numpy as np
import random
import sys


def get_text_list(fname):
    text_names = []
    f = open(fname, "r", encoding='ascii')
    lines = f.readlines()
    for line in lines:
        text_names.append(line[:-1])
    f.close()
    return text_names


def data_preprocessing(fname, isTest=False):
    # Fetch texts from local txt file
    text = ""
    f = open(fname, "r", encoding='ascii')
    lines = f.readlines()
    for line in lines:
        if line != "":
            text += line[:-1]
    f.close()

    # Tokenization
    sentences = sent_tokenize(text.lower())  # lower all strings
    token_sentences = [word_tokenize(s) for s in sentences]
    padded_sentences = [list(pad_both_ends(s, n=2)) for s in token_sentences]

    if isTest:
        return padded_sentences, None
    else:
        n_dev = len(padded_sentences) // 10
        random.shuffle(padded_sentences)
        test_sentences = padded_sentences[:n_dev]
        train_sentences = padded_sentences[n_dev:]
        # train_text = list(flatten(train_sentences))
        # test_text = list(flatten(test_sentences))
        return train_sentences, test_sentences


def test_data_preprocessing(fname):
    sentences = []
    f = open(fname, "r", encoding='ascii')
    lines = f.readlines()
    for line in lines:
        sentences.append(line[:-1].lower())
    f.close()
    token_sentences = [word_tokenize(s) for s in sentences]
    padded_sentences = [list(pad_both_ends(s, n=2)) for s in token_sentences]
    return padded_sentences


def train(train_text):
    train, vocab = padded_everygram_pipeline(3, train_text)
    lm = KneserNeyInterpolated(3, discount=0.75)
    lm.fit(train, vocab)
    return lm


def evaluate_single(lm, s):
    test_trigrams = list(trigrams(s))
    return lm.perplexity(test_trigrams)


def evaluate_multiple(lm, test_sentences):
    test_trigrams = []
    for s in test_sentences:
        test_trigrams.extend(list(trigrams(s)))
    return lm.perplexity(test_trigrams)


def main():
    random.seed(7)

    isTest = None
    author_menu_path = None
    test_menu_path = None

    argv = sys.argv[1:]
    print(f"### HW2 for CSCI 5541 23S ###")
    print(f"### Author: Bin Hu (BH's Group) ###")
    if len(argv) == 1:
        author_menu_path = argv[0] + ".txt"
        isTest = False
    elif len(argv) == 3 and argv[1] == '-t':
        author_menu_path = argv[0] + ".txt"
        test_menu_path = argv[2]
        isTest = True
    else:
        print("invalid input, loading the default set")
        author_menu_path = "authorlist.txt"
        isTest = False

    nltk.download('punkt')
    author_names = get_text_list(author_menu_path)

    if isTest:
        data = defaultdict(dict)

        for name in author_names:
            data[name]['train_set'], _ = data_preprocessing(name, False)

        print("training LMs... (this may take a while)")
        for name in author_names:
            data[name]['lm'] = train(data[name]['train_set'])

        test_sentences = test_data_preprocessing(test_menu_path)

        for s in test_sentences:
            lowest_perplexity = float('inf')
            result = 'error'
            for name in author_names:
                if evaluate_single(data[name]['lm'], s) < lowest_perplexity:
                    lowest_perplexity = evaluate_single(data[name]['lm'], s)
                    result = name
            print(result[:-4])

    else:
        data = defaultdict(dict)

        print("splitting into training and development...")
        for name in author_names:
            data[name]['train_set'], data[name]['test_set'] = data_preprocessing(name, False)

        print("training LMs... (this may take a while)")
        for name in author_names:
            data[name]['lm'] = train(data[name]['train_set'])

        print("Results on dev set:")
        for name in author_names:
            result = []
            for s in data[name]['test_set']:
                self_perplexity = evaluate_single(data[name]['lm'], s)
                all_perplexities = []
                for arbi_name in author_names:
                    all_perplexities.append(evaluate_single(data[arbi_name]['lm'], s))
                result.append(1.0 if abs(self_perplexity - min(all_perplexities)) < 0.000001 else 0)
                # if the sentence's perplexity is the least on the lm trained on its author's text
                # classification is correct
            precision = np.asarray(result).mean()
            perplexity_on_test = evaluate_multiple(data[name]['lm'], data[name]['test_set'])
            print("{}:   \t{:.2f}% correct\t{:.2f} perplexity".format(name[:-4], precision * 100, perplexity_on_test))


if __name__ == '__main__':
    main()
