import pickle

with open('./models/gridsearch.pkl', 'rb') as handle:
    gs_dict = pickle.load(handle)
