import numpy as np
import plotly

plotly.tools.set_credentials_file(username='IdoKessler', api_key='hGdHmrAHFW5CcyBj8BDz')

import plotly.plotly as py
import plotly.graph_objs as go

import fill_missing_values
from sklearn.decomposition import PCA
from data_utils import *

def reduceDataTo3features(X):
    assert len(X.shape) == 3, 'expect 3d shapes'
    X = X.reshape(X.shape[0], -1)
    return PCA(n_components=3).fit_transform(X)

print ("Parsing: ")
X, Y = get_X_Y_train()
X = fill_missing_values.fill_with_mean(X)
X = reduceDataTo3features(X)


x = X[:, 0]
y = X[:, 1]
z = X[:, 2]
c = Y[:, 0]

trace1 = go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=12,
        color=c,                # set color to an array/list of desired values
        colorscale='Viridis',   # choose a colorscale
        opacity=0.8
    )
)

data = [trace1]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)

fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename='3d-scatter-colorscale')
