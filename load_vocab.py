import sys
import json
import read_from_txt

def load_path_index(path):
    lines = read_from_txt.read_from_split(path)
    res = {}
    index = 0
    for data in lines:
        cur = data[0].strip()
        if cur not in res and cur!='':
            res[cur] = index
            index +=1
    return res

def load_path_value(path, split = '\t'):
    lines = read_from_txt.read_from_split(path,split=split)
    res = {}
    for data in lines:
        cur = data[0].strip()
        if cur not in res and cur!='':
            res[cur] = data[1]
    return res


