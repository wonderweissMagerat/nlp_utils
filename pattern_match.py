import re
import json

from load_vocab import load_path_value

def load_pattern_cate(path,split = ' '):
    pattern_dict = load_path_value(path, split=split)
    return pattern_dict



def match_sentence(pattern, sentence):
    res = False
    txt = None
    match = re.search(pattern, sentence)
    if match:
        return True,match
    else:
        return res,txt


if __name__ == '__main__':
    pattern = 'dog'
    sentence = 'aadogggg'
    res,sent = match_sentence(pattern, sentence)
    print(sent.group())



