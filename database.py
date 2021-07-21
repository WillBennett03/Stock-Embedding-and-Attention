from pymongo import MongoClient

client = MongoClient("mongodb+srv://WBennett03:mK96Dco67cCbZitx@cluster0.j4bhe.mongodb.net/myFirstDatabase?retryWrites=true&w=majority")
db = client.myFirstDatabase

db.record.insert_one({'TEST' : {'sect' : 'meme', 'data' : [[1,1],[1,2],[1,3]]}})