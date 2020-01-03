# from Rucio traces extract all the paths accessed by MWT2 ANALY jobs not running via VP.
# store all the data in hdf5 file

#!bin/python

import time
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan

es = Elasticsearch(hosts=['http://atlas-kibana.mwt2.org:9200'], http_auth=('xxxx', 'xxx'))
print(es.ping())


query = {
    "size": 0,
    "_source": ["scope", "dataset", "filename", "timeStart", "filesize", "pq"],
    "query": {
        "bool": {
            "must": [
                {"term": {"event_type": "get_sm_a"}},
                {"term": {"pq": "ANALY_MWT2_UCORE"}}
            ]
        }
    }
}

data = []

es_response = scan(es, index='rucio_traces',  query=query)
count = 0
for item in es_response:
    sou = item['_source']
    doc = [
        sou['scope'],
        sou['dataset'],
        sou['filename'],
        sou['timeStart'],
        sou['filesize']
    ]
    print(doc)
    data.append(doc)
    break

    if count and not count % 1000:
        print(count)
    count += 1

print(count)
