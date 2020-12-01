import sys
import json

def read_from_split(path, split='\t'):
    res = []
    for lines in open(path):
        data = lines.strip('\n').split('\t')
        res.append(data)
    return res

def read_from_json(path,key = {}):
    res = []
    for lines in open(path):
        data = json.loads(lines.strip())
        res.append(res)
    return res

