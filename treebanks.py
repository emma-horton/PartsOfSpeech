from io import open

from conllu import parse_incr

treebank = {}
treebank['en'] = 'treebanks/UD_English-LinES/en_lines'
treebank['sv'] = 'treebanks/UD_Swedish-LinES/sv_lines'
treebank['ko'] = 'treebanks/UD_Korean-GSD/ko_gsd'

def train_corpus(lang):
    return treebank[lang] + '-ud-train.conllu'

def test_corpus(lang):
    return treebank[lang] + '-ud-test.conllu'

# Remove contractions such as "isn't".
def prune_sentence(sentence):
    return [token for token in sentence if type(token['id']) is int]

def conllu_corpus(path):
    data_file = open(path, 'r', encoding='utf-8')
    sentences = list(parse_incr(data_file))
    return [prune_sentence(sentence) for sentence in sentences]
