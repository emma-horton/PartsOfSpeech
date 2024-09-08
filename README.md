# Parts of Speech Tagging with Dynamic Algorithms

![Python Badge](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=flat)
![NLTK Badge](https://img.shields.io/badge/NLTK-3776AB?logo=nltk&logoColor=white&style=flat)
![CoNLLU Badge](https://img.shields.io/badge/CoNLLU-3776AB?logo=conllu&logoColor=white&style=flat)
![Dynamic Programming Badge](https://img.shields.io/badge/Dynamic_Programming-3776AB?logo=dynamicprogramming&logoColor=white&style=flat)

## Overview
This project was developed for the “Language and Computation” module at the University of St Andrews as an individual assignment. The program, written in Python, utilizes data collected from UD Treebanks. The data was processed using the CoNLL-U package, and the model was implemented with the help of NLTK.

## Purpose 
To implement and compare the Viterbi algorithm with the Eager and Individually Most Probable Tags algorithms to optimize part of speech tagging accuracy.

## Features 
* Implementation of three algorithms with varying complexity: Eager, Viterbi, and Individually Most Probable Tags.
* Training on corpora in three distinct languages: English, Swedish, and Korean.
* Evaluation of part-of-speech tagging accuracy for each language using unseen test sets.

## Usage
#### 1. Install dependancies 
```bash
pip install conullu
npm install nltk
```
#### 2. Run script
```bash
python3 p1.py
```

## Technologies Used 
* **CoNLL-U**: Utilized for parsing and organizing corpora from the UD Treebank into training and testing sets.
* **NLTK**: Employed to compute emission and transition probabilities essential for training the models.

## Acknowledgements
* [UD Treebank](https://universaldependencies.org): Credited for supplying the data used to train and test the models across various languages.


## Introduction
Part of speech (POS) tagging is a process in computational linguistics where each word in a sentence is assigned a label that indicates its grammatical role, such as noun, verb, adjective etc. These labels are crucial for understanding the syntax and structure of sentences. It is important for tasks such as speech generation, information extraction, parsing and machine translation. The purpose of this assignment was to gain an understanding of the Viterbi algorithm, and its application to POS tagging. The aim was to implement and compare the performance of the Viterbi algorithm with two related algorithms, the Eager and Individually Most Probable Tags. Their performance in three languages, English, Korean and Swedish was compared to derive findings about the languages morphological and syntactic properties.

## Data Preparation
Data was collected from the UD Treebanks. It was parsed and split into training and test sets using the conllu python package. Training data was formatted into a 2D list of sentences containing words with associated tags. Each sentence was tagged with a start and end of sentence marker to capture boundary effects in the transition probability estimations. Transmission probabilities were calculated by counting relative frequency of POS bigrams with Witten bell smoothing. Emission probabilities was calculated by counting the relative frequency of a given word associated with a POS. Finally, start of sentence probability was calculated by counting the relative frequency of a given POS at the start of the sentence.

## Algorithms and Implementation
### Eager Algorithm
The Eager algorithm represents a straightforward approach to POS tagging. It operates on a token-by-token basis, selecting for each token the most probable tag based on the preceding tokens tag and current token itself. In the script, the Eager algorithm is encapsulated within the ‘eager algorithm’ function. For each word in the input sentence, the algorithm iterates through all possible tags, calculates the probability of each tag by multiplying the emission probability of the word given the tag with the transition probability from the previous tag to the current tag, and selects the tag with the highest probability.

### Viterbi Algorithm
The Viterbi algorithm is a more complex approach that considers the entire sentence context. It dynamically computes the most probable sequence of tags for a given sentence. In the script, the Viterbi algorithm is implemented in the ‘viterbi_algorithm’ function. The function initialises the V matrix with log probabilities for the first word across all tags based on the start probabilities and emission probabilities. It then iterates over each word in the sentence, updating the matrix based on the transition probabilities, emission probabilities, and previously computed log probabilities. Finally, it backtracks from the last word to determine the most probable sequence of tags. To handle the numerical instability issues, such as underflow due to the multiplication of small probabilities, log probabilities are used. By transforming probabilities into log space, multiplication operations become additions, significantly mitigating the underflow risk. This approach is critical for the Viterbi algorithm's reliability, especially in processing long sentences where the product of many small probabilities would otherwise approach zero.
  
### Individually Most Probable Tags
The individually most probable tags algorithm was the most sophisticated approach, focusing on maximising the probability of a correct POS for each word given the context of the other tags in the sentence. In the script, it is implemented in the ‘most_probable_tags_algorithm function. It calls two auxiliary functions, the ‘forward_algorithm’ and ‘backward_algorithm’ functions which return a matrix of the forward and backward probabilities of each word being a given tag. The `most_probable_tags_algorithm` function works in several steps to determine the most likely tag for each word in a sentence. First, it calculates the sum of the forward probabilities, which are probabilities moving from the beginning of the sentence up to the previous work, considering both emission probabilities and transition probabilities. Then, it computes the sum of the backward probabilities, which are probabilities moving from the end of the sentence back to the next word, again considering emission and transition probabilities. The logsumexp() function provided was used to sum probabilities accurately. These two sums are then multiplied together. The tag that results in the highest probability for a given word is selected as the most probable tag.

## Algorithm Accuracy Evaluation

The accuracy of the algorithms was evaluated by comparing the predicted tags against the true tags from the test set. Accuracy was calculated as the percentage of words correctly tagged by each algorithm for each given language. The results are summarized in the table below:

| Language | Accuracy % Eager | Accuracy % Viterbi | Accuracy % Individually Most Probable Tags |
|----------|------------------|--------------------|-------------------------------------------|
| English  | 88.6             | 91.3               | 88.6                                      |
| Swedish  | 85.7             | 90.2               | 85.7                                      |
| Korean   | 80.8             | 79.2               | 80.8                                      |

**Figure 1** – Table of accuracy scores for each algorithm in each given language.

The accuracy from the Individually Most Probable Tags algorithm was not evaluated in the discussion as it did not produce a higher accuracy than Viterbi as expected.

## Discussion

### Comparing the performance of the two algorithms

#### Viterbi Algorithm had higher accuracy than the Eager Algorithm for Swedish and English but a slightly lower accuracy for Korean.
Both English and Swedish benefit from the Viterbi algorithm's ability to consider the entire sentence context. These languages adhere to relatively strict syntactic rules where the order of words significantly influences their grammatical roles. However, Korean's agglutinative nature means that a lot of grammatical information is encoded within individual word forms through various affixes. This rich morphological information can often provide strong cues for the POS of a word independently of its broader sentence context. Furthermore, Korean's syntax allows for a considerable degree of flexibility in word order, which can diminish the utility of analysing sentence-wide patterns for POS tagging.

#### Eager Algorithm had relabvely high accuracy for all languages.
The UD Treebank is a relabvely coarse treebank as it only contains 17 tags for each language. This is because tags have been grouped into universal categories instead of being language specific. As a result, with fewer tags to choose from, the probability of correctly guessing the tag for a given word increases. This can make a simpler model like Eager Algorithm more effecbve as it does not need to make such fine morphological disbncbons. This increases the emission probability producing a given tag and creates more clear transibon probabilibes between tags.

### Comparing the performance in the three languages

#### English saw the highest accuracy in the Viterbi and Eager Algorithm, followed by Swedish and Korean.
English has simpler morphology compared to Swedish and Korean. This means that there are fewer inflected and affixed words to recognise and characterise, making the tagging process easier. Additionally, its relatively fixed syntactic structure means that the sequence of words in a sentence is more predictable, which both algorithms can exploit effectively.

#### Swedish saw the greatest increase in accuracy from the Eager to the Viterbi algorithm.
Swedish, while not as morphologically rich as Korean, has more inflection than English and allows for more variation in sentence structure. The Viterbi algorithm's ability to analyse the entire sentence structure can significantly benefit languages with these characteristics, as it can better model the dependencies between words and tags over longer distances than the Eager algorithm.

### Comparing the difficulty of POS tagging for different languages
In this practical Korean saw significantly lower accuracy in POS tagging than English and Swedish. This may be because Korean is an agglutinative language meaning that it forms words and expresses grammatical relationships through extensive use of affixes. As a result, this morphological complexity creates a vast number of possible word forms from a single root, complicating the POS tagging process. In contrast, English and Swedish are analytical languages that has simpler morphology, meaning that there are fewer forms of each word to recognise and categorise, making the tagging process easier.

### Reflecting on challenges encountered and how they were addressed

#### Viterbi Algorithm
Building the Viterbi algorithm was challenging as it involved deep understanding of the underlying mathematical logic to transform the formulas learnt in class to python code. Writing the function took multiple iterations of development, reflecting on content taught from the lecture and referencing the textbook to look at pseudocode.

#### Individually Most Probable Parts of Speech Algorithm
Implementing the logic behind Individually Most Probable Parts of Speech Algorithm was challenging as this content was not directly covered in class or the textbook. Required deep thought about mathematical underpinning to understand the formulas and formulate an approach to integrating the forward and backward probabilities into a predictive algorithm. Further iterations of development may be required to refine this algorithm to get expected results.
