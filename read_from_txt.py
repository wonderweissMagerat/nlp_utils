import sys
import json

def read_from_split(path, split='\t'):
    res = []
    for lines in open(path):
        data = lines.strip('\n').split(split)
        res.append(data)
    return res

def read_from_json(path,loc=0,split='\t'):
    res = []
    for lines in open(path):
        data = json.loads(lines.strip().split(split)[loc])
        res.append(data)
    return res

