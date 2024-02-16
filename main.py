import pickle
import random

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from numpy.linalg import svd
from pydantic import BaseModel

# App creation and model loading

app = FastAPI()


class Response(BaseModel):
    ans: list


class Rec:

    def __init__(self):
        self.df, self.df_data, self.dict_of_beer, self.df_matrix, self.vh = self.initialization()

    def initialization(self):
        df = pd.read_pickle('df')
        df_data = pd.read_pickle('df_data')
        dict_of_beer = pd.read_pickle('dict_of_beer.pkl')
        df_matrix = df_data.pivot_table(index="review_profilename", columns="beer_beerid", values="review_overall",
                                        aggfunc='mean').fillna(0)
        vh = np.load('svd_for_items.npy')
        return df, df_data, dict_of_beer, df_matrix, vh

    def searching(self, tmp):
        for el in tmp:
            if self.dict_of_beer.get(el):
                return el

    def cosine_similarity(self, v, u):
        return (v @ u) / (np.linalg.norm(v) * np.linalg.norm(u))

    def rec_cos(self, name: str, vh) -> dict:
        data = pd.DataFrame.from_dict(self.dict_of_beer, orient='index')
        highest_similarity = -np.inf
        highest_sim_col = -1
        id_beer = []
        sim = []
        tmp = self.df.loc[self.df["beer_name"] == name, "beer_beerid"]
        el = self.searching(tmp)
        for col in range(1, self.vh.shape[1]):
            similarity = self.cosine_similarity(self.vh[:, data.index.get_loc(el)], vh[:, col])
            id_beer.append(self.df_matrix.columns[col])
            sim.append(similarity)
        return dict(zip(id_beer, sim))

    def data_processing(self, data, top_n):
        top = []
        for el in data.split():
            sorted_ = sorted(self.rec_cos(el.replace('_', ' '), self.vh).items(), key=lambda x: x[1], reverse=True)
            for index in range(top_n):
                top.append(self.dict_of_beer[sorted_[index][0]][0])
        random.shuffle(top)
        return top[:top_n]


model = Rec()


@app.get('/recommend')
def recommend(data: str):
    top = model.data_processing(data, 7)

    response = Response(
        ans=top
    )

    return response


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
