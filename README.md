![Python Badge](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=flat)
![NLTK Badge](https://img.shields.io/badge/NLTK-3776AB?logo=nltk&logoColor=white&style=flat)
![CoNLLU Badge](https://img.shields.io/badge/CoNLLU-3776AB?logo=conllu&logoColor=white&style=flat)
![Dynamic Programming Badge](https://img.shields.io/badge/Dynamic_Programming-3776AB?logo=dynamicprogramming&logoColor=white&style=flat)
# Parts of Speech Tagging with Dynamic Algorithms
Implementing three part-of-speech tagging algorithms—Eager, Viterbi, and Individually Most Probable Tags—and comparing their accuracy across English, Korean, and Swedish.
## Purpose 
This project was developed as an individual assignment as part of the coursework for the “Language and Computation” at the University of St Andrews. The three algoritms were trained using data collected from [Universal Dependancies Treebank](https://universaldependencies.org). Python, along with the CoNLL-U package and NLTK, were used to process the data and train the algorithms.

As an extension of the assignment, an interactive interface was developed to visualise the inner workings of the Viterbi algorithm, which achieved test set accuracies of 91.3% in English, 90.2% in Swedish, and 79.2% in Korean.
<figure>
    <img src="images/POSinterface" alt="POSInterface" width="1000">
<!--     <figcaption>Part of Speech Tagging Interface</figcaption> -->
</figure>

## Aims
* Develop and implement three algorithms of varying complexity: Eager, Viterbi, and Individually Most Probable Tags.
* Train the algorithms on corpora from three distinct languages: English, Swedish, and Korean.
* Evaluate the part-of-speech tagging accuracy for each language using unseen test sets.

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
* [Universal Dependancies Treebank](https://universaldependencies.org): Provided the multilingual data used for training and testing the models.
  
## Project Grade 
Grade: 17/20 (85%)
