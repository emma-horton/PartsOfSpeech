from flask import Flask, render_template, request
from treebanks import conllu_corpus, train_corpus, test_corpus
from nltk.probability import WittenBellProbDist, FreqDist
from sys import float_info
from math import log
from nltk.util import bigrams
from langdetect import detect
import json
import os
# Add other necessary imports from your script

app = Flask(__name__)

# Define your POS tagging logic as a function
def pos_tagger(test_sentence, lang):
    # Your existing code with minimal modifications goes here
    # Choose language.
    user_test_sentence = test_sentence

    # Create train and test set
    train_sents = conllu_corpus(train_corpus(lang))

    # Processing Training Sentences
    tagged_sentences = []
    for sent in train_sents:
        s = [(token['form'], token['upos']) for token in sent]
        tagged_sentences.append(s)

    # Calculating transition probabilties
    transition_counts = []
    for sentence in tagged_sentences:
        # add markers for start and end of sentence 
        tags = ['<s>'] + [tag for word, tag in sentence] + ['</s>']
        transition_counts.append(list(bigrams(tags)))

    transition_counts_flat =[]
    for sentence in transition_counts:
        transition_counts_flat.extend(sentence)
        
    varbigrams = FreqDist(transition_counts_flat)
    transition_probabilities = WittenBellProbDist(varbigrams, bins=1e5)

    # Calculating emission probabilties
    emission_probabilities = {}
    emission_counts_flat =[]
    for sentence in tagged_sentences:
        emission_counts_flat.extend(sentence)
    tags = set([t for (_,t) in emission_counts_flat])
    for tag in tags:
        words = [w for (w,t) in emission_counts_flat if t == tag]
        emission_probabilities[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)

    # Calculating start probabilities
    start_tag_counts = {'ADJ': 0, 'ADP': 0, 'ADV':0, 'AUX':0, 'CCONJ':0, 'DET':0, 'INTJ':0, 'NOUN':0, 'NUM':0, 'PART':0, 'PRON':0, 'PROPN':0, 'PUNCT':0, 'SCONJ':0, 'SYM':0, 'VERB':0, 'X':0 }
    for sentence in tagged_sentences:
        start_tag = sentence[0][1] 
        start_tag_counts[start_tag] += 1
    total_sentences = len(tagged_sentences)
    num_tags = len(start_tag_counts)
    alpha = 1
    start_p = {tag: (count + alpha) / (total_sentences + alpha * num_tags) for tag, count in start_tag_counts.items()}



    # Define your Viterbi algorithm function
    def viterbi_algorithm(obs, states, start_p, trans_p, emit_p):
        # Your Viterbi algorithm code here
        V = [{}]
        path = {}
        backpointer = [{}]

        # Initial state probabilities
        for st in states:
            V[0][st] = log(start_p[st]) + log(emit_p[st].prob(obs[0]))
            backpointer[0][st] = None

        for t in range(1, len(obs)):
            V.append({})
            newpath = {}
            backpointer.append({})
            for st in states:
                (max_log_prob, prev_st) = max(
                    (V[t-1][prev] + log(trans_p.prob((prev, st))) + log(emit_p[st].prob(obs[t])), prev)
                    for prev in states
                )
                V[t][st] = max_log_prob
                backpointer[t][st] = prev_st

        last_state = max(V[-1], key=V[-1].get)
        best_path = []
        t = len(obs) - 1
        while last_state is not None:
            best_path.insert(0, last_state)
            last_state = backpointer[t][last_state]
            t -= 1

        return best_path, V, backpointer

    user_words = user_test_sentence.split()
    tag_set = list(emission_probabilities.keys())
    user_prediction_viterbi, V, backpointer = viterbi_algorithm(user_words, tag_set, start_p, transition_probabilities, emission_probabilities)
    return user_prediction_viterbi, V, backpointer 

@app.route('/', methods=['GET', 'POST'])
def index():
    output = []
    test_sentence = ''
    vTable = []
    backpointer = {}
    if request.method == 'POST':
        test_sentence = request.form.get("sentence")
        lang = request.form.get("language")
        detected_lang = detect(test_sentence)
        print(f'detected lang {detected_lang}')
        if detected_lang == lang:
            output, vTable, backpointer = pos_tagger(test_sentence, lang)
    return render_template('index.html', output=output, sentence=test_sentence, vTable=vTable, backpointer=backpointer)


if __name__ == "__main__":
    # Set debug mode to True when running locally
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5001)), debug=True)
