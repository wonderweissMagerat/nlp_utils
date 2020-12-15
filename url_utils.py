import requests

def post_url(url,info):
    return_info = requests.post(url,json = info).text
    return return_info
