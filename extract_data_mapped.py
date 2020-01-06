# from Rucio traces extract all the paths accessed by MWT2 ANALY jobs not running via VP.
# store all the data in hdf5 file

from time import time
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import pandas as pd
from secret import es_auth

pq = 'ANALY_MWT2_UCORE'
# pq = 'ALL'

es = Elasticsearch(hosts=['http://atlas-kibana.mwt2.org:9200'], http_auth=es_auth)
print(es.ping())

dt = 30 * 86400
query = {
    "size": 0,
    "_source": ["scope", "dataset", "filename", "timeStart", "filesize"],
    "query": {
        "bool": {
            "must": [
                {"term": {"eventType": "get_sm_a"}},
                {"term": {"pq": pq}},
                {"range": {"timeStart": {"gt": int(time() - dt), "format": "epoch_second"}}},
                {"exists": {"field": "filesize"}}
            ]
        }
    }
}

data = []
count = 0

scopes = {}
datasets = {}
filenames = {}

cur_sco = 0
cur_dat = 0
cur_fil = 0

# es_response = scan(es, index='rucio-traces-2020*',  query=query)
es_response = scan(es, index='rucio_traces', query=query, request_timeout=60)
for item in es_response:
    sou = item['_source']

    if sou['scope'] in scopes.keys():
        scope = scopes[sou['scope']]
    else:
        scopes[sou['scope']] = cur_sco
        scope = cur_sco
        cur_sco += 1

    if sou['dataset'] in datasets.keys():
        dataset = datasets[sou['dataset']]
    else:
        datasets[sou['dataset']] = cur_dat
        dataset = cur_dat
        cur_dat += 1

    if sou['filename'] in filenames.keys():
        filename = filenames[sou['filename']]
    else:
        filenames[sou['filename']] = cur_fil
        filename = cur_fil
        cur_fil += 1

    doc = [scope, dataset, filename, sou['timeStart'], sou['filesize']]
#     print(doc)
    data.append(doc)

    if count and not count % 1000:
        print(count)
    count += 1

print(count)

accesses = pd.DataFrame(data).sort_values(4)
accesses.columns = ['scope', 'dataset', 'filename', 'timeStart', 'filesize']
# accesses.set_index('filename', drop=True, inplace=True)
accesses.to_hdf('data/' + pq + '_m.h5', key='data', mode='w', complevel=1)

scopes_map = pd.DataFrame.from_dict(scopes, orient='index')
scopes_map.to_hdf('data/' + pq + '_m.h5', key='scopes_map', mode='a', complevel=1)

datasets_map = pd.DataFrame.from_dict(datasets, orient='index')
datasets_map.to_hdf('data/' + pq + '_m.h5', key='datasets_map', mode='a', complevel=1)

filenames_map = pd.DataFrame.from_dict(filenames, orient='index')
filenames_map.to_hdf('data/' + pq + '_m.h5', key='filenames_map', mode='a', complevel=1)
