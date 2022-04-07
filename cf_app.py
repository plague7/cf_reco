import os
import os.path
import streamlit as st

import pandas as pd
import seaborn as sns

from fastai.collab import *
from fastai.tabular.all import *

#CSV loading#
dossiers_path = os.path.join(os.path.dirname('__file__'), 'dossiers.csv')
ratings_path = os.path.join(os.path.dirname('__file__'), 'ratings.csv')

dossiers = pd.read_csv(dossiers_path)
ratings = pd.read_csv(ratings_path)

#dossiers = pd.read_csv("C:/Users/Simplon/Desktop/Alternance Efalia/dossiers.csv")
#ratings = pd.read_csv("C:/Users/Simplon/Desktop/Alternance Efalia/ratings.csv")

#DF preprocessing#

df = ratings.merge(dossiers)
df = df.fillna(0)

def basic_prez():
    global df_prez_base
    df_prez_base = df.pivot(index='user', columns='nom', values='score')

#CollabNN#
class CollabNN(Module):
    def __init__(self, user_sz, item_sz, y_range=(0,5.5), n_act=100):
        self.user_factors = Embedding(*user_sz)
        self.item_factors = Embedding(*item_sz)
        self.layers = nn.Sequential(
            nn.Linear(user_sz[1]+item_sz[1], n_act),
            nn.ReLU(),
            nn.Linear(n_act, 1))
        self.y_range = y_range
        
    def forward(self, x):
        embs = self.user_factors(x[:,0]),self.item_factors(x[:,1])
        x = self.layers(torch.cat(embs, dim=1))
        return sigmoid_range(x, *self.y_range)

#Functions#
def make_dls():
    global embs, df_prez, dls
    dls = CollabDataLoaders.from_df(df, item_name='nom', bs=6)
    embs = get_emb_sz(dls)

def create_empty_model():
    global embs, dls, learn
    model = CollabNN(*embs)
    learn = Learner(dls, model, loss_func=MSELossFlat())

def filter_lr():
    global learn, best_lr
    lr_value = str(learn.lr_find())
    splitting = str(lr_value).split('=')
    almost_there = splitting[1]
    best_lr = float(almost_there[0:10])

def model_train():
    global best_lr
    learn.fit_one_cycle(5, best_lr, wd=0.01)

def predict():
    global to_predict
    to_predict = df.loc[df['score'] == 0.0]
    to_predict_dl = learn.dls.test_dl(to_predict)
    preds, _, decoded = learn.get_preds(dl=to_predict_dl, with_decoded=True)
    idx = to_predict.index.tolist()
    for i in range(1,(len(idx))+1):
        j=i-1
        pred = (preds.numpy()[j]).item()
        to_predict.iat[j,2] = pred

def new_value():
    global to_predict, df_copie, df_final
    df_copie = df.copy()
    df_copie.loc[:, ['score']] = to_predict[['score']]
    df_copie = df_copie.fillna(0)
    df_final = df.copy()

    df_final['score'] = (df['score']+df_copie['score'])

def pivoting():
    global df_copie, df_final, df_prez, df_copie_prez, df_final_prez
    df_prez = df.pivot(index='user', columns='nom', values='score')
    df_copie_prez = df_copie.pivot(index='user', columns='nom', values='score')
    df_final_prez = df_final.pivot(index='user', columns='nom', values='score')

## STREAMLIT FUNCTIONS##

st.title('Modele de recommendation "Collaborative Filtering"')

#Selectbox#

choice = st.selectbox('Database', ['Modele de base','J\'importe mes donnees'])
tableau = st.selectbox('Tableau', ['Tableau rempli', 'Juste les predictions', 'Tableau original'])


if choice == 'Modele de base':
    basic_prez()
    st.dataframe(df_prez_base)
    if st.button('Voir les predictions pour les cellules vides'):
        make_dls()
        create_empty_model()
        filter_lr()
        model_train()
        predict()
        new_value()
        pivoting()
        st.write('Choisissez quel tableau vous souhaitez visualisez')
        if tableau == 'Tableau rempli':
            st.dataframe(df_final_prez)
        elif tableau == 'Juste les predictions':
            st.dataframe(df_copie_prez)
        elif tableau == 'Tableau original':
            st.dataframe(df_prez)
elif choice == 'J\'importe mes donnees':
    st.write('Work in progress')
