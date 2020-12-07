import sys
import json

def write_from_dict_split(path, jdata={},mode='w', split='\t'):
    output = open(path,mode)
    for jd in jdata:
        output.write(json.dumps(jd)+split+json.dumps(jdata[jd]))
        output.write('\n')
