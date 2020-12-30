import sys
import json
import read_from_txt

def load_path_index(path):
    lines = read_from_txt.read_from_split(path)
    res = {}
    index = 0
    for data in lines:
        if data[0] not in res and data[0]!='':
            res[data[0]] = index
            index +=1
    return res
