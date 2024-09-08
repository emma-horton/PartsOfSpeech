from treebanks import conllu_corpus, train_corpus, test_corpus
from nltk.probability import WittenBellProbDist, FreqDist
from sys import float_info
from math import log, exp
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from nltk.util import bigrams

#################################################### DATA PREPARATION #####################################################################
# Choose language.
lang = 'sv'

# create train and test set 
train_sents = conllu_corpus(train_corpus(lang))
test_sents = conllu_corpus(test_corpus(lang))

#Processing Training Sentences
tagged_sentences = []
for sent in train_sents:
	s = []
	for token in sent:
		s.append((token['form'], token['upos']))
	tagged_sentences.append(s)
 
# getting testing tags 
testing_tags = []
for sent in test_sents:
	for token in sent:
		testing_tags.append(token['upos'])
 

print('language', lang)
print(len(train_sents), 'training sentences')
print(len(test_sents), 'test sentences')

#################################################### PARAMETER ESTIMATION #####################################################################

# Calculating transition probabilties
transition_counts = []
for sentence in tagged_sentences:
    # add markers for start and end of sentence 
    tags = ['<s>'] + [tag for word, tag in sentence] + ['</s>']
    transition_counts.append(list(bigrams(tags)))

transition_counts_flat =[]
for sentence in transition_counts:
    transition_counts_flat.extend(sentence)
    
bigrams = FreqDist(transition_counts_flat)
transition_probabilities = WittenBellProbDist(bigrams, bins=1e5)

# Calculating emission probabilties
emission_probabilities = {}
emission_counts_flat =[]
for sentence in tagged_sentences:
    emission_counts_flat.extend(sentence)
tags = set([t for (_,t) in emission_counts_flat])
for tag in tags:
    words = [w for (w,t) in emission_counts_flat if t == tag]
    emission_probabilities[tag] = WittenBellProbDist(FreqDist(words), bins=1e5)
print('Smoothed emission probability of INTJ to Yes is', emission_probabilities['INTJ'].prob('Yes'))

# Calculating start probabilities
start_tag_counts = {'ADJ': 0, 'ADP': 0, 'ADV':0, 'AUX':0, 'CCONJ':0, 'DET':0, 'INTJ':0, 'NOUN':0, 'NUM':0, 'PART':0, 'PRON':0, 'PROPN':0, 'PUNCT':0, 'SCONJ':0, 'SYM':0, 'VERB':0, 'X':0 }
for sentence in tagged_sentences:
    start_tag = sentence[0][1] 
    start_tag_counts[start_tag] += 1
total_sentences = len(tagged_sentences)
num_tags = len(start_tag_counts)
alpha = 1
start_p = {tag: (count + alpha) / (total_sentences + alpha * num_tags) for tag, count in start_tag_counts.items()}

#################################################### ALOGORITHM 1 #####################################################################
# Implmeneting the Eager Algorithm 
def eager_algorithm(sentence, emission_probs, transition_probs):
    tagged_sentence = []  # This will store the resulting POS tagged sentence
    prev_tag = '<s>'  # Start-of-sentence tag, this corresponds to t0 in the formula

    # Loop through each word in the sentence
    for word in sentence:
        max_prob = 0  # Initialise the maximum probability for the current word
        best_tag = None  # Initialise the best tag found for the current word
    
        # Loop through each possible tag in the emission probabilities
        for tag in emission_probs:
            # Calculate the emission probability of the word given the tag
            emission_prob = emission_probs[tag].prob(word)
            # Calculate the transition probability from the previous tag to the current tag
            transition_prob = transition_probs.prob((prev_tag, tag))
            # The overall probability for the current tag is the product of the emission and transition probabilities
            prob = emission_prob * transition_prob

            # If the calculated probability is greater than the max_prob, update max_prob and best_tag
            if prob > max_prob:
                max_prob = prob
                best_tag = tag

        # Append the best tag to the tagged_sentence list
        tagged_sentence.append(best_tag)
        prev_tag = best_tag  # Update the tag of the previous word to be used in the next iteration
    
    return tagged_sentence  # Return the POS tagged sentence

# The function is implementing the eager algorithm for POS tagging.
# It chooses the POS tag for the i-th token based on the previous tag and the current word.
# This corresponds to the formula t_i = argmax_t P(t | t_{i-1}) * P(w_i | t_i)
# where t_{i-1} is the tag of the previous word (prev_tag in the code),
# and t_i is the tag for the current word.

    
#################################################### ALOGORITHM 2 #####################################################################    

def viterbi_algorithm(obs, states, start_p, trans_p, emit_p):
    V = [{}]  # This corresponds to the Viterbi table viterbi[q, i]
    path = {}
    backpointer = [{}]  # The backpointer table used for backtracking the best path

    # Initialise base cases (t == 0)
    # This is the initialisation step which corresponds to viterbi[q, 1] = α(q0, q) * B(q, w1)
    # Here, q0 is the start state and B(q, w1) is the emission probability of state q for the first observation w1.
    for st in states:
        V[0][st] = log(start_p[st]) + log(emit_p[st].prob(obs[0]))
        backpointer[0][st] = None  # No backpointer for the first state

    # Run Viterbi for t > 0
    # This is the recursion step. For each state st at time t, find the state that maximises the
    # probability of the path ending in st. This corresponds to the equation:
    # viterbi[q, i] = max_q' viterbi[q', i-1] * α(q', q) * B(q, wi), where q' is a previous state.
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}
        backpointer.append({})

        for st in states:
            # Choose the previous state with the highest transition probability and emission probability 
            (max_log_prob, prev_st) = max(
                (V[t-1][prev] + log(trans_p.prob((prev, st)))+ log(emit_p[st].prob(obs[t])), prev) for prev in states
            )
            # The current Viterbi probability is the maximum log probability from the previous state
            V[t][st] = max_log_prob
            # Record the backpointer for the current state st at time t
            backpointer[t][st] = prev_st

    # Identify the final state with the maximum probability
    # This corresponds to the termination step of the Viterbi algorithm where we identify the state
    # with the highest probability at the final time step.
    last_state = max(V[-1], key=V[-1].get)

    # Backtrack to reconstruct the path
    # Starting from the last state, follow the backpointers to find the most probable path that led to the final state
    best_path = []
    t = len(obs) - 1
    while last_state is not None:
        best_path.insert(0, last_state)
        last_state = backpointer[t][last_state]
        t -= 1

    return best_path


#################################################### ALGORITHM 3 #####################################################################
def forward_algorithm(obs, states, start_p, trans_p, emit_p):
    forward = [{}]
    
    for st in states:
        forward[0][st] = log(start_p[st]) + log(emit_p[st].prob(obs[0]))
    
    for t in range(1, len(obs)):
        forward.append({})
        for st in states:
            # log sum rather than max function 
            forward[t][st] = logsumexp([forward[t-1][prev_st] + log(trans_p.prob((prev_st, st))) + log(emit_p[st].prob(obs[t]))
                                         for prev_st in states])
    return forward

def backward_algorithm(obs, states, start_p, trans_p, emit_p):
    n = len(obs)
    backward = [{} for _ in range(n + 1)]
    
    for st in states:
        backward[n][st] = log(1)
    # iterate in opposite direction 
    for t in reversed(range(n)):
        for st in states:
            backward[t][st] = logsumexp(
                [log(trans_p.prob((st, next_st))) + log(emit_p[next_st].prob(obs[t])) + backward[t + 1][next_st]
                 for next_st in states]
            )
    return backward

            
def most_probable_tags_algorithm(obs, states, start_p, trans_p, emit_p):
    forward = forward_algorithm(obs, states, start_p, trans_p, emit_p)
    backward = backward_algorithm(obs, states, start_p, trans_p, emit_p)
    best_path = []

    # Pre-calculate cumulative sums for forward and backward probabilities
    for_cum = [sum(forward[i][st] for st in states) for i in range(len(obs))]
    back_cum = [sum(backward[i][st] for st in states) for i in range(len(obs)-1, -1, -1)]
    back_cum.reverse()  # Reverse to align with forward probabilities order

    prev = '<s>'  # Initialise previous tag for the first iteration
    for i in range(len(obs)):
        f_b_total = {
            st: (for_cum[i] + log(emit_p[st].prob(obs[i])) + log(trans_p.prob((prev, st)))) +
                (back_cum[i] + log(emit_p[st].prob(obs[i])) + log(trans_p.prob((prev, st))))
            for st in states
        }

        most_probable_tag = max(f_b_total, key=f_b_total.get)
        best_path.append(most_probable_tag)
        prev = most_probable_tag  # Update previous tag for next iteration

    return best_path
   

# Adding a list of probabilities represented as log probabilities.
min_log_prob = -float_info.max
def logsumexp(vals):
	if len(vals) == 0:
		return min_log_prob
	m = max(vals)
	if m == min_log_prob:
		return min_log_prob
	else:
		return m + log(sum([exp(val - m) for val in vals]))
#################################################### EVALUATION #####################################################################

# Apply Algorithms to the Test Sentences
prediction_eager = []
prediction_viterbi =[]
prediction_most_probable = []
for test_sent in test_sents:
    words = [token['form'] for token in test_sent]  
    tagged_sentence_eager = eager_algorithm(words, emission_probabilities, transition_probabilities)
    prediction_eager.extend(tagged_sentence_eager)

    tag_set = list(emission_probabilities.keys())
    tagged_sentence_viterbi = viterbi_algorithm(words, tag_set, start_p, transition_probabilities, emission_probabilities)
    prediction_viterbi.extend(tagged_sentence_viterbi)
    
    tagged_sentence_eager_most_probable = most_probable_tags_algorithm(words, tag_set, start_p, transition_probabilities, emission_probabilities)
    prediction_most_probable.extend(tagged_sentence_eager_most_probable)


# Calculate accuracy
print('Evaluation on test data')
accuracy_eager = accuracy_score(testing_tags, prediction_eager)
accuracy_viterbi = accuracy_score(testing_tags, prediction_viterbi)
accuracy_most_probable = accuracy_score(testing_tags, prediction_most_probable)

print(f"Accuracy (Eager Algorithm): {accuracy_eager * 100:.2f}%")
print(f"Accuracy (Viterbi Algorithm): {accuracy_viterbi * 100:.2f}%")
print(f"Accuracy (Most Probable Algorithm): {accuracy_most_probable * 100:.2f}%")



# Find all unique tags to ensure consistency
unique_tags_eager = sorted(set(testing_tags + prediction_eager))
unique_tags_viterbi = sorted(set(testing_tags + prediction_viterbi))

confusion_matrix_eager = confusion_matrix(testing_tags, prediction_eager, labels=unique_tags_eager)
confusion_matrix_viterbi = confusion_matrix(testing_tags, prediction_viterbi, labels=unique_tags_viterbi)


# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_eager, display_labels=unique_tags_eager)
disp.plot(cmap=plt.cm.Blues)
plt.title("Eager Algorithm") 
plt.xticks(rotation=45) 
plt.show()

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_viterbi, display_labels=unique_tags_viterbi)
disp.plot(cmap=plt.cm.Blues)
plt.title("Viterbi Algorithm") 
plt.xticks(rotation=45) 
plt.show()
