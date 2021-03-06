[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/AlrxKali/AB_testing_Mobile_game">
    <img src="img/LogoSample_ByTailorBrands (1).jpg" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">A/B Testing a mobile game</h3>

  <p align="center">
    Alejandro Alemany
    <br />
    <a href="https://github.com/AlrxKali/AB_testing_Mobile_game"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/AlrxKali/AB_testing_Mobile_game">View Demo</a>
    ·
    <a href="https://github.com/AlrxKali/AB_testing_Mobile_game">Report Bug</a>
    ·
    <a href="https://github.com/AlrxKali/AB_testing_Mobile_game">Request Feature</a>
  </p>
</p>

[license-shield]: img/license.svg
[license-url]: https://github.com/AlrxKali/AB_testing_Mobile_game/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/alejandro-alemany/

# Cookie Cats 
<span>
<p align="center">
  <a href="https://github.com/AlrxKali/AB_testing_Mobile_game">
    <img src="img/download.jpg" alt="Logo" width="180" height="180" align="center">
  </a>
</span>
<br>

##### Cookie Cats is a popular mobile game with more than 1M downloads and more than 95,000 reviews. The game is a "connect three" style puzzle game. As the players level up, they will find gates that force them to wait a time, watch an ad, or purchase to continue playing. In this project, we want to analyze the impact on player "retention" of moving the first gate from 30 to 40.

<span>
<p align="right">
  <a href="https://github.com/AlrxKali/AB_testing_Mobile_game">
    <img src="img/Screenshot.jpg" alt="Logo" width="480" height="280" align="left">
  </a>
</span>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>
<br>

The data was collected by Tactile Entertainment, and it provides the following variables:

1.   userid - A unique number that identifies each player.
2.   version - Whether the player was put in the control group (gate_30 - a gate at level 30) or the group with the moved gate (gate_40 - a gate at level 40).
3.   sum_gamerounds - The number of game rounds played by the player during the first 14 days after install.
3.   retention_1 - Did the player come back and play 1 day after installing?
4.   retention_7 - Did the player come back and play 7 days after installing?

<br>

# Exploratory Data Analysis

The data has 90189 values organized in 5 columns.


```python
print(f'Table properties: {data.shape}')
print('_______________________________')
print(f'Values types: \n{data.dtypes}')
print('_______________________________')
print(f'General description of statistics: \n{data.describe()}')
```

    Table properties: (90189, 5)
    _______________________________
    Values types: 
    userid             int64
    version           object
    sum_gamerounds     int64
    retention_1         bool
    retention_7         bool
    dtype: object
    _______________________________
    General description of statistics: 
                 userid  sum_gamerounds
    count  9.018900e+04    90189.000000
    mean   4.998412e+06       51.872457
    std    2.883286e+06      195.050858
    min    1.160000e+02        0.000000
    25%    2.512230e+06        5.000000
    50%    4.995815e+06       16.000000
    75%    7.496452e+06       51.000000
    max    9.999861e+06    49854.000000
    

The goal of this project is to find a better way to increase retention. So far, the hypothesis is that moving the first gate from level 30 to level 40 will increase retention. However, **how much do they play?**


```python
plot_df = data.groupby('sum_gamerounds').count()

ax = plot_df[:50].iloc[:, :3].plot()
ax.set_title("Sum of Game Rounds after 1 day")
ax.set_ylabel("Count of players")
plt.show()
```


![png](img/output_9_0.png)


The graph below shows that after 20 rounds, the number of users that continue engaged falls below 1000. The game rounds do not increase after seven days, either. 


```python
ax = plot_df[:50].plot()
ax.set_title("Sum of Game Rounds after 1 day")
ax.set_ylabel("Count of players")
plt.show()
```


![png](img/output_11_0.png)


Both graphs seem identical at first sight. It is clear that game rounds fall after 20 matches, so it is crucial to see whether changing the first gate will help to increase retention. 


```python
gate_30 = data[data['version'] == 'gate_30']
gate_40 = data[data['version'] == 'gate_40']

bins = [0,1,10,20,30,40,50,60,70,80,90,100,200,500]
bins_gate_30 = pd.DataFrame(gate_30.groupby(pd.cut(gate_30["sum_gamerounds"], bins=bins)).count())
bins_gate_40 = pd.DataFrame(gate_40.groupby(pd.cut(gate_40["sum_gamerounds"], bins=bins)).count())
```


```python
ax = bins_gate_30[:50].plot(kind = 'bar', y="userid", color = "black", 
                       title = 'Total Usage By Groups')
bins_gate_40[:50].plot(kind = 'bar', y="userid", ax=ax, color = "green", alpha = 0.7 )
ax.set_xlabel("Total Game Rounds")
ax.set_ylabel("Players")
plt.legend(["gate_30", "gate_40"])
plt.grid(True)
```


![png](img/output_14_0.png)


There is a slight difference in both groups, so it will be interesting to find how significant is the difference. There is an increase of game rounds when the first gate is at 30, but **how significant is the difference?**


```python
gate_30_percent = round(data[data['version'] == 'gate_30']['retention_1'].mean(), 4)
gate_40_percent = round(data[data['version'] == 'gate_40']['retention_1'].mean(), 3)

print(f'Retention percentage at gate 30: {gate_30_percent * 100}%')
print(f'Retention percentage at gate 40: {gate_40_percent * 100}%')
print(f'Retention difference: {round(gate_30_percent - gate_40_percent, 4)*100}%')
```

    Retention percentage at gate 30: 44.82%
    Retention percentage at gate 40: 44.2%
    Retention difference: 0.62%
    

Percent of retention is almost the same. However, there is a 0.62% difference, and that difference could mean an increase in earning from ads and conversion. By plotting the distributions of gate_30 and gate_40, it is clear that there is a big difference in their means. **However, how confident can we be that by moving the gate from 30 to 40 will bring a loss of 0.62%?**


```python
distributions = []

for _ in range(1000):
  dist_mean = data.retention_1.sample(frac=1, replace=True).groupby(data.version).mean()
  distributions.append(dist_mean)
```


```python
distributions = pd.DataFrame(distributions)
plt.hist(distributions['gate_30'], alpha = 0.5)
plt.hist(distributions['gate_40'], alpha = 0.5)
plt.show()
```


![png](img/output_19_0.png)



```python
g = sns.boxenplot(data=[distributions['gate_30'],
                        distributions['gate_40']])
                        
g.set(xticklabels = ['gate_30', 'gate_40'])
plt.show()
```


![png](img/output_20_0.png)


The data is close to half for gate_30 and half for gate_40. However, the information we need to test the hypothesis is categorical. Therefore, a Chi-Square will be the tool used for the analysis.


```python
g = sns.pointplot(data=[distributions['gate_30'],
                        distributions['gate_40']])
                        
g.set(xticklabels = ['gate_30', 'gate_40'])
plt.show()
```


![png](img/output_22_0.png)



```python
pd.crosstab(data['version'], data['retention_1'], margins = True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>retention_1</th>
      <th>False</th>
      <th>True</th>
      <th>All</th>
    </tr>
    <tr>
      <th>version</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>gate_30</th>
      <td>24666</td>
      <td>20034</td>
      <td>44700</td>
    </tr>
    <tr>
      <th>gate_40</th>
      <td>25370</td>
      <td>20119</td>
      <td>45489</td>
    </tr>
    <tr>
      <th>All</th>
      <td>50036</td>
      <td>40153</td>
      <td>90189</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Getting the observations and expected
observed_values = pd.crosstab(data['version'], data['retention_1'])
contingency = stats.chi2_contingency(observed_values)
expected_values = contingency[3]

print(f'Observed values: \n{observed_values.values}\n')
print(f'Expected values: \n{expected_values}')
```

    Observed values: 
    [[24666 20034]
     [25370 20119]]
    
    Expected values: 
    [[24799.13514952 19900.86485048]
     [25236.86485048 20252.13514952]]
    


```python
# Finding degree of freedom
num_rows = len(observed_values)
num_cols = len(observed_values)
dof = (num_rows - 1) * (num_cols - 1)
alpha = 0.05

print(f'Degree of freedom: {dof}')
```

    Degree of freedom: 1
    

Now that we have set the alpha, and we have found the degree of freedom, observed, and expected values, it is time to compute the **Chi-Square**

![png](img/download1.png)


```python
def chi2_stats(observed, expected):
  chi_square = sum([(o - e)**2./e for o,e in zip(observed, expected)])
  chi_square_statistics = chi_square[0] + chi_square[1]

  print(f'Chi-Square Statistics: {chi_square_statistics}')
  return chi_square_statistics
```


```python
chi2_statistics = chi2_stats(observed_values.values, expected_values)
```

    Chi-Square Statistics: 3.182963657512031
    


```python
critical_value = chi2.ppf(q=1-alpha, df = dof)
critical_value
```




    3.841458820694124




```python
# Finding the P-Value
p_value = 1 - chi2.cdf(x=chi2_statistics, df=dof)
print(f'p-value: {p_value}')
print(f'significance level: {alpha}')
print(f'degree of freedom: {dof}')
```

    p-value: 0.0744096552969219
    significance level: 0.05
    degree of freedom: 1
    


```python
if chi2_statistics >= critical_value:
  print("Reject H0, there is a relationship between two categorical variables")
else:
  print("Retain H0, there is no a relationship between two categorical variables")

if p_value <= alpha:
  print("Reject H0, there is a relationship between two categorical variables")
else:
  print("Retain H0, there is no a relationship between two categorical variables")
```

    Retain H0, there is no a relationship between two categorical variables
    Retain H0, there is no a relationship between two categorical variables
    

According to the result obtained from the Chi-square, the gates are not related to retention. However, we have seen more rounds of games at gate 30 than at level 40. 

The probability of retention at each level might help to clarify the situation.


```python
# Finding the difference in the retention distribution
distributions["difference"] = distributions.gate_30 - distributions.gate_40
```


```python
plt.hist(distributions['difference'], alpha = 0.5, color= 'green')
plt.axvline(x=distributions['difference'].mean(), color='red')
plt.axvline(x=0.0, color='blue')
plt.show()
```


![png](img/output_35_0.png)


Most of the distribution under the curve is positive. Therefore, the retention percent is higher at gate 30. However, **what is the probability that the retention will be higher after one day of installing the game?**


```python
# Finding the probability that retention at gate 30 will be higher
prob = (distributions['difference']>0).sum()/len(distributions)

print(f'Probability of that the retention will be higher at gate 30 after 1 day since downloading the game: {prob*100}%')
```

    Probability of that the retention will be higher at gate 30 after 1 day since downloading the game: 96.8%
    

# Analyzing retention after 7 days

After 1 day since installing the game, retention is higher at gate 30. We want to know if the condition is the same after 7 days. 


```python
gate_30_percent = round(data[data['version'] == 'gate_30']['retention_7'].mean(), 4)
gate_40_percent = round(data[data['version'] == 'gate_40']['retention_7'].mean(), 3)

print(f'Retention percentage at gate 30: {gate_30_percent * 100}%')
print(f'Retention percentage at gate 40: {gate_40_percent * 100}%')
print(f'Retention difference: {round(gate_30_percent - gate_40_percent, 3)*100}%')
```

    Retention percentage at gate 30: 19.02%
    Retention percentage at gate 40: 18.2%
    Retention difference: 0.8%
    

After 7 days, retention has fallen even more, but the difference is two point higher.


```python
t1 = pd.crosstab(data['version'], data['retention_7'], margins = True)
t1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>retention_7</th>
      <th>False</th>
      <th>True</th>
      <th>All</th>
    </tr>
    <tr>
      <th>version</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>gate_30</th>
      <td>36198</td>
      <td>8502</td>
      <td>44700</td>
    </tr>
    <tr>
      <th>gate_40</th>
      <td>37210</td>
      <td>8279</td>
      <td>45489</td>
    </tr>
    <tr>
      <th>All</th>
      <td>73408</td>
      <td>16781</td>
      <td>90189</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Getting the observations and expected
observed_values = pd.crosstab(data['version'], data['retention_7'])
contingency = stats.chi2_contingency(observed_values)
expected_values = contingency[3]

print(f'Observed values: \n{observed_values.values}\n')
print(f'Expected values: \n{expected_values}')
```

    Observed values: 
    [[36198  8502]
     [37210  8279]]
    
    Expected values: 
    [[36382.90257127  8317.09742873]
     [37025.09742873  8463.90257127]]
    


```python
# Finding degree of freedom
num_rows = len(observed_values)
num_cols = len(observed_values)
dof = (num_rows - 1) * (num_cols - 1)
alpha = 0.05

print(f'Degree of freedom: {dof}')
```

    Degree of freedom: 1
    


```python
chi2_statistics = chi2_stats(observed_values.values, expected_values)
```

    Chi-Square Statistics: 10.013167328688969
    


```python
critical_value = chi2.ppf(q=1-alpha, df = dof)
critical_value
```




    3.841458820694124




```python
# Finding the P-Value
p_value = 1 - chi2.cdf(x=chi2_statistics, df=dof)
print(f'p-value: {p_value}')
print(f'significance level: {alpha}')
print(f'degree of freedom: {dof}')
```

    p-value: 0.0015542499756142636
    significance level: 0.05
    degree of freedom: 1
    


```python
if chi2_statistics >= critical_value:
  print("Reject H0, there is a relationship between two categorical variables")
else:
  print("Retain H0, there is no a relationship between two categorical variables")

if p_value <= alpha:
  print("Reject H0, there is a relationship between two categorical variables")
else:
  print("Retain H0, there is no a relationship between two categorical variables")
```

    Reject H0, there is a relationship between two categorical variables
    Reject H0, there is a relationship between two categorical variables
    

There is a relationship between retention and the version of the game after 7 days. Therefore, with a higher retention percent, we have enough evidence to keep the gate at level 30

To see how strong the relationship is between version and retention after 7 days. I will conduct an effect size test. 


```python
from statsmodels.stats.power import TTestIndPower

analysis = TTestIndPower()
```

    /usr/local/lib/python3.6/dist-packages/statsmodels/tools/_testing.py:19: FutureWarning: pandas.util.testing is deprecated. Use the functions in the public API at pandas.testing instead.
      import pandas.util.testing as tm
    


```python
result = analysis.solve_power(effect_size=None, power=0.8, nobs1=t1.iloc[2,2], ratio=1.0, alpha=0.05)
print('The effect size needed for this experiment would be {}'.format(result))
```

    The effect size needed for this experiment would be 0.013192984981012151
    

It was expected that moving the first gate from 30 to level 40 will increase retention. However, the analysis shows that it is not the case. Graphs showed that people started to lose interest in the game after 20 rounds. People who reach level 30, and are forced to stop, have a more significant motivation to come back later or make a purchase to continue playing. However, by the time a user reaches level 40, interest might have decrease.
