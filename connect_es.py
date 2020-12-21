from elasticsearch import Elasticsearch

es = Elasticsearch(hosts=['http://nonlocal-es.ha.nb.com:9200'])
#delete index
#es.indices.delete(index='category')
#print(es.cat.indices())
index = 'cluster-doc-*'
query_json = {
  "query": {
    "bool": {
      "must": [
        {"regexp": {
          "title": "Affair"
        }}
      ]
    }
  }
}
res = es.search(index = index,body=query_json)
print(res['hits']['total'])
for hit in res['hits']['hits']:
    print(hit)



