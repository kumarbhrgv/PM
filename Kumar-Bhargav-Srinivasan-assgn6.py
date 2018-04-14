%env MKL_THREADING_LAYER=GNU
import pymc3 as pm
import math
import matplotlib.pyplot as plt

with pm.Model() as dna_model:
    g1 = pm.Bernoulli('g1', .5)
    g2 = pm.Bernoulli('g2', pm.math.switch(g1 > 0, 0.1, 0.9))
    g3 = pm.Bernoulli('g3', pm.math.switch(g1 > 0, 0.1, 0.9))
    var_1 = pm.math.switch(g1 > 0, 50, 60)
    var_2 = pm.math.switch(g2 > 0, 50, 60)
    var_3 = pm.math.switch(g3 > 0, 50, 60)
    x1 = pm.Normal('x1', var_1, math.sqrt(10))
    x2 = pm.Normal('x2', var_2, math.sqrt(10),observed=50)
    x3 = pm.Normal('x3', var_3, math.sqrt(10))

with dna_model:
    step = pm.Metropolis()
    start = pm.find_MAP()
    trace = pm.sample(50000,random_seed=123)#,start=start,step=step,random_seed=123)
    df = pm.trace_to_dataframe(trace=trace)
    print(len(df))
    pm.traceplot(trace)
    plt.show()
    trace_g1 = df['g1'].values
    count = 0
    for i in trace_g1:
        if round(i) == 1:
            count += 1
    probability_conditional = count/float(len(trace_g1))
    print(probability_conditional)
    trace_x3 = df['x3'].values
    count = 0
    for i in trace_x3:
        if round(i) == 50:
            count += 1
    probability_conditional = count/float(len(trace_x3))
    print(probability_conditional)
