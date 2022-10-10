import pickle

client = {"reports": 0, "share": 0.001694, "expenditure": 0.12, "owner": "yes"}

with open('dv.bin', 'rb') as dict_in:
    dict_vectorizer = pickle.load(dict_in)
    X = dict_vectorizer.transform(client)
    # print("dict file opened")
    with open('model1.bin', 'rb') as model_in:  
        model = pickle.load(model_in)
        # print("model file opened")
        pred = model.predict_proba(X)
        print(pred[0,1])
    model_in.close()
dict_in.close()


