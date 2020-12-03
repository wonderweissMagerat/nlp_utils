import sys
import json
from read_from_txt import read_from_json


def analyse_tf_distribution_json_file(path,key='',value=''):
    data = read_from_json(path)
    res = {}
    for jd in  data:
        cur_key  = jd[key]
        cur_value = jd[value]
        cur_res = res.get(cur_key,{})
        cur_res_value = cur_res.get(cur_value,0)
        cur_res[cur_value] = cur_res_value+1
        res[cur_key] = cur_res[cur_value]
    for k in res:
        cur = list(sorted(res[k].items(),key = lamda x:x[1],reverse = True))
        res[k] = cur
    return res



