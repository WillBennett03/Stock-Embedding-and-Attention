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


def embed_sp500(embedding_size = 5, epochs=25):
    INDEX = ign.index_data('SP500')


    embed = model.get_embedding_model(embedding_size, INDEX.stock_vocab_size, INDEX.time_vocab_size, 5)
    model.train_model(embed, INDEX.x, INDEX.y, epochs, 'base_3')
    graphing.show_embedding(embed, 'models/base_3.ckpt', INDEX.sect)

def get_sp500_data():
    train_json = ign.build_index_dict('SP500', save=True)

def get_embedding_map(embedding_size = 50):
    train_json = ign.get_dict('data/SP500.json')
    x, y, sect, stock_vocab_size, time_vocab_size = ign.preprocess_dict(train_json)
    embed = model.get_embedding_model(embedding_size, stock_vocab_size, time_vocab_size, 5)
    graphing.show_embedding(embed, 'models/base_3.ckpt', sect)


if __name__ =='__main__':
    embed_sp500()   
    # get_embedding_map() 
