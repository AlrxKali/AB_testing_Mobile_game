import plotly.offline as pyo
import plotly.graph_objs as go
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('cookie_cats.csv')
rounds = data.groupby('sum_gamerounds')['userid'].count().reset_index()
trace = go.Scatter(x=rounds.iloc[:, 0][:50], y=rounds.iloc[:, 1])

data = [trace]

layout = go.Layout(title='Line chart test')

fig = go.Figure(data=data, layout=layout)
pyo.plot(fig)
