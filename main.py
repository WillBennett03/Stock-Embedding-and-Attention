"""
###############
### Main.py ###
###############

~ Will Bennett 11/07/2021

This is the main file :D 10 points if you already worked that out
"""
import ingestion as ign
import model 
import graphing


def embed_sp500(embedding_size = 5, epochs=2500):
    train_json = ign.get_dict('data/SP500.json')
    x, y, sect, stock_vocab_size, time_vocab_size = ign.preprocess_dict(train_json)
    print((x.shape, y.shape))

    print(stock_vocab_size)
    print(time_vocab_size)
    embed = model.get_embedding_model(embedding_size, stock_vocab_size, time_vocab_size, 5)
    model.train_model(embed, x, y, epochs, 'base_3')
    graphing.show_embedding(embed, 'models/base_3.ckpt', sect)

def control_embed_sp500(embedding_size = 50, epochs=100):
    train_json = ign.get_dict('data/SP500.json')
    x, y, sect, stock_vocab_size, time_vocab_size = ign.preprocess_dict(train_json)
    print((x.shape, y.shape))

    print(stock_vocab_size)
    print(time_vocab_size)
    embed = model.get_control_embedding_model(embedding_size, stock_vocab_size, 5)
    model.train_control_embedding_model(embed, x, y, epochs, 'base_3')
    graphing.show_control_embedding(embed, 'models/base_3.ckpt', sect)


def get_sp500_data():
    train_json = ign.build_index_dict('SP500', save=True)

def get_embedding_map(embedding_size = 50):
    train_json = ign.get_dict('data/SP500.json')
    x, y, sect, stock_vocab_size, time_vocab_size = ign.preprocess_dict(train_json)
    embed = model.get_embedding_model(embedding_size, stock_vocab_size, time_vocab_size, 5)
    graphing.show_embedding(embed, 'models/base_3.ckpt', sect)

if __name__ =='__main__':
    # embed_sp500()   
    get_embedding_map() 
