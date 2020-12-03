import sys
import json

def write_from_dict_split(path, jdata={},keys=[],mode='w', split='\t'):
    output = open(path,mode)
    for jd in jdata:
        for k in keys:
            output.write(json.dumps(k)+split+json.dumps(jd[k])+split)
        output.write('\n')
