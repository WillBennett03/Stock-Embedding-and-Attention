from pymongo import MongoClient
url = ''

client = MongoClient(url)
db = client.myFirstDatabase

db.record.insert_one({'TEST' : {'sect' : 'meme', 'data' : [[1,1],[1,2],[1,3]]}})
