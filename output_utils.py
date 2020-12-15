import sys
import json
import read_from_txt
def write_from_dict_split(path, jdata={},mode='w', split='\t'):
    output = open(path,mode)
    for jd in jdata:
        output.write(json.dumps(jd)+split+json.dumps(jdata[jd]))
        output.write('\n')

def json_to_split(path,output_path,key=[],loc = 0):
    data = read_from_txt.read_from_json(path,loc = loc)
    output = open(output_path,'w')
    for jd in data:
        for k in key:
            output.write(str(jd.get(k))+'\t')
        output.write('\n')