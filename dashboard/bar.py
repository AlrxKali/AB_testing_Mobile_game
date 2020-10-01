import plotly.offline as pyo
import plotly.graph_objs as go
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('cookie_cats.csv')
gate_30 = data[data['version'] == 'gate_30']
gate_40 = data[data['version'] == 'gate_40']

bins = [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500]
bins_gate_30 = pd.DataFrame(gate_30.groupby(
    pd.cut(gate_30["sum_gamerounds"], bins=bins)).count())
bins_gate_40 = pd.DataFrame(gate_40.groupby(
    pd.cut(gate_40["sum_gamerounds"], bins=bins)).count())

fig = go.Figure(data=[
    go.Bar(
        name='gate_30',
        y=bins_gate_30.iloc[:, 0][:50],
    ),
    go.Bar(
        name='gate_40',
        y=bins_gate_40.iloc[:, 0][:50],
        marker_color='lightsalmon'
    )
])

pyo.plot(fig)
