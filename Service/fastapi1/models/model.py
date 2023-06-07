import pickle




def model():

    model = pickle.load(open('/home/bmf/Desktop/freelance/fastapi1/models/LGBM.pickle', 'rb'))
    return model


