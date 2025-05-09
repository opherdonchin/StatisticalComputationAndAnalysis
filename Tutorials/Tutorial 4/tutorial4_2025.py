# -*- coding: utf-8 -*-
"""Tutorial4_2025.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1AW7HARbRBP9gAq50Z9XtoFCNZE_vp4rI

# **Tutorial 4**

## **Imports**
"""

import numpy as np # arrays, array operations
import scipy.stats as stats # statistics
from google.colab import files
import matplotlib.pyplot as plt # plot graphs
import pandas as pd #dataframes
import io
import xarray as xr #multidimensional dataframes

#preliz
!pip install preliz

import preliz as pz

#installs for reading the google sheets
!pip install --upgrade gspread
!pip install --upgrade oauth2client
!pip install --upgrade gspread google-auth oauth2client

#authentication
from google.colab import auth
from google.auth.transport.requests import Request
import gspread
from google.auth import default

# Authenticate the user
auth.authenticate_user()

# Get the credentials
creds, _ = default()

# Authorize gspread with the credentials
gc = gspread.authorize(creds)

#getting our email collected data
# Open the spreadsheet by its ID
spreadsheet_id = '1N5I487HjDrl1Ep76ruVv0MvVpjL1cHRL6BBeHZU_9zg'  # Replace with your own spreadsheet ID
worksheet = gc.open_by_key(spreadsheet_id).sheet1

# Read all records into a DataFrame
data = worksheet.get_all_records()
df = pd.DataFrame(data)

# Display the DataFrame
df.head()

data = df['How many emails do you receive each day?'].to_numpy()
print(data)

n = len(data)
print(f'There are {n} participants')

S = np.sum(data)
print(f'There are a total of {S} emails')

#simulate data
data = np.random.poisson(8, 24)
print(data)

n = len(data)
print(f'There are {n} participants')

S = np.sum(data)
print(f'There are a total of {S} emails')

#look at the data
plt.hist(data, bins = 10, align = 'left')
plt.xlabel('Number of Emails')
plt.ylabel('Counts')
plt.savefig("data.png", bbox_inches='tight')
files.download("data.png")

"""## PyMC"""

import pymc as pm
import arviz as az

coords = {"data": np.arange(n)}

with pm.Model(coords = coords) as our_first_model:
    lambda_ = pm.Gamma('lam', alpha = 1.68, beta = 0.0569)
    k = pm.Poisson('k', mu = lambda_, observed=data, dims = 'data')
    idata = pm.sample(1000, chains = 4)

"""### Inference Data Object"""

idata

"""**Posterior**"""

#we can save an xarray of the posterior information
posterior = idata.posterior.lam
posterior.shape #and see that the shape is 1000 draws for 4 chains

#we can also convert the information into a numpy array
posterior_np = posterior.to_numpy()
posterior_np

"""Analyzing the Posterior"""

az.plot_trace(idata, compact = False)

plt.savefig("trace.png", bbox_inches='tight')
files.download("trace.png")

#comparing with our analytical results
alpha = 1.68
beta = 0.0569
x = np.linspace(0, 100, 2000)

posterior2 = pz.Gamma(alpha + S, beta + n).pdf(x)
plt.plot(x, posterior2)
plt.xlim([6.5, 11])

plt.savefig("posterior_analytical.png", bbox_inches='tight')
files.download("posterior_analytical.png")

"""Means of Chains"""

ms = idata.posterior.mean(dim = 'draw')
print(f'The means of the 4 chains are: {np.round(ms.lam.to_numpy(), 3)}')

"""Effect of number of samples (draws)

"""

with pm.Model(coords = coords) as our_second_model:
    lambda_ = pm.Gamma('lam', alpha = 1.68, beta = 0.0569)
    k = pm.Poisson('k', mu = lambda_, observed=data, dims = 'data')
    idata2 = pm.sample(2000, chains = 4, random_seed=4591)

az.plot_trace(idata2, compact = False)

plt.savefig("trace2.png", bbox_inches='tight')
files.download("trace2.png")

ms = idata2.posterior.mean(dim = 'draw')
print(f'The means of the 4 chains are: {np.round(ms.lam.to_numpy(), 3)}')

with pm.Model(coords = coords) as our_third_model:
    lambda_ = pm.Gamma('lam', alpha = 1.68, beta = 0.0569)
    k = pm.Poisson('k', mu = lambda_, observed=data, dims = 'data')
    idata3 = pm.sample(10000, chains = 4, random_seed=4591)

az.plot_trace(idata3, compact = False)

plt.savefig("trace3.png", bbox_inches='tight')
files.download("trace3.png")

ms = idata3.posterior.mean(dim = 'draw')
print(f'The means of the 4 chains are: {np.round(ms.lam.to_numpy(), 3)}')

#for computing modes
def kde_mode(values):
    kde = stats.gaussian_kde(values)
    x = np.linspace(values.min(), values.max(), 1000)
    mode = x[np.argmax(kde(x))]
    return mode

mode_values = xr.apply_ufunc(
    lambda x: kde_mode(x),  # Apply KDE mode computation
    posterior,               # The xarray DataArray
    input_core_dims=[["draw"]],  # Compute mode along "draw"
    vectorize=True  # Ensure compatibility with multiple dimensions
)

print(mode_values)

#combining the 4 KDEs
az.plot_trace(idata, combined = True)

m = idata.posterior.mean()
print(f'The total mean of the 4000 samples is: {np.round(m.lam.to_numpy(), 3)}')

#we can also get all the data as 4000 samples using extract, instead of as four seperate chains
az.extract(idata)

#if you specify number of samples, you get random sampling out of the 4000
az.extract(idata, num_samples = 3500)

"""HDI"""

#plotting the posterior with the HDI
az.plot_posterior(idata)


plt.savefig("hdi.png", bbox_inches='tight')
files.download("hdi.png")

#and getting the values in a table
az.summary(idata, kind = 'stats').round(2)



"""Savage-Dickey Density Ratio"""

az.plot_bf(idata, var_name="lam", prior = np.random.gamma(1.68, 1/0.0569, 10000), ref_val = 9)

plt.savefig("sd.png", bbox_inches='tight')
files.download("sd.png")

"""ROPE"""

az.plot_posterior(idata, rope=[8, 10])

plt.savefig("rope.png", bbox_inches='tight')
files.download("rope.png")

"""## PyTensor

Some of the following is taken from the PyMC notebook: https://www.pymc.io/projects/docs/en/stable/learn/core_notebooks/pymc_pytensor.html
"""

import pytensor
import pytensor.tensor as pt

"""To begin, we define some pytensor tensors and show how to perform some basic operations.


"""

x = pt.scalar(name="x")
y = pt.vector(name="y")

print(
    f"""
x type: {x.type}
x name = {x.name}
---
y type: {y.type}
y name = {y.name}
"""
)

"""Now that we have defined the x and y tensors, we can create a new one by adding them together.


"""

z = x + y
z.name = "x + y"
print(f'z type: {z.type}\nz name = {z.name}')

"""To make the computation a bit more complex let us take the logarithm of the resulting tensor.


"""

w = pt.log(z)
w.name = "log(x + y)"

"""We can use the dprint() function to print the computational graph of any given tensor.


"""

pytensor.dprint(w)

"""Note that this graph does not do any computation (yet!). It is simply defining the sequence of steps to be done. We can use function() to define a callable object so that we can push values trough the graph."""

f = pytensor.function(inputs=[x, y], outputs=w)
pytensor.dprint(f)

"""Now that the graph is compiled, we can push some concrete values:


"""

f(x=0, y=[1, np.e])

"""One of the most important features of pytensor is that it can automatically optimize the mathematical operations inside a graph. Let’s consider a simple example:"""

a = pt.scalar(name="a")
b = pt.scalar(name="b")

c = a / b
c.name = "a / b"

pytensor.dprint(c)

"""Now let us multiply b times c. This should result in simply a.


"""

d = b * c
d.name = "b * c"

pytensor.dprint(d)

"""The graph shows the full computation, but once we compile it the operation becomes the identity on a as expected."""

g = pytensor.function(inputs=[a, b], outputs=d)

pytensor.dprint(g)

"""Random variables in PyTensor"""

y = pt.random.normal(loc=0, scale=1, name="y")
y.type

#look at the graph
pytensor.dprint(y)

"""We could sample by calling eval(). on the random variable.


"""

for i in range(10):
    print(f"Sample {i}: {y.eval()}")

"""We always get the same samples! This has to do with the random seed step in the graph, i.e. RandomGeneratorSharedVariable. We will show how to generate different samples with pymc below.

To do so, we start by defining a pymc normal distribution.
"""

x = pm.Normal.dist(mu=0, sigma=1)
pytensor.dprint(x)

for i in range(10):
    print(f"Sample {i}: {x.eval()}")

"""As before we get the same value for all iterations. The correct way to generate random samples is using draw()."""

#and plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(pm.draw(x, draws=1_000), color="C1", bins=15)
ax.set(title="Samples from a normal distribution using pymc", ylabel="count")

"""### What is going on behind the scenes?

We can now look into how this is done inside a Model.
"""

with pm.Model() as model:
    z = pm.Normal(name="z", mu=0, sigma=1)

pytensor.dprint(z)

"""We are just creating random variables like we saw before, but now registering them in a pymc model. To extract the list of random variables we can simply do:"""

pytensor.dprint(model.basic_RVs)

for i in range(10):
    print(f"Sample {i}: {pm.draw(z)}")

"""### Log Probabilities
(these are often used to help achieve numeric stability in the sampling algorithms)

pymc is able to convert RandomVariables to their respective probability functions. One simple way is to use logp(), which takes as first input a RandomVariable, and as second input the value at which the logp is evaluated.
"""

z_value = pt.vector(name="z")
z_logp = pm.logp(rv=z, value=z_value)

pytensor.dprint(z_logp)

"""Observe that, as explained at the beginning, there has been no computation yet. The actual computation is performed after compiling and passing the input. For illustration purposes alone, we will again use the handy eval() method."""

z_logp.eval({z_value: [0]})

"""This is nothing else than evaluating the log probability of a normal distribution.

### Derivatives and Gradients
"""

x = pt.scalar('x')        # Symbolic scalar input
f = x**2                  # Define the function f(x) = x^2

pytensor.dprint(f)

"""Compute the derivative"""

df_dx = pytensor.grad(f, x)   # Compute derivative df/dx
pytensor.dprint(df_dx)

"""We can evaluate for specific values"""

f_func = pytensor.function([x], f)
df_func = pytensor.function([x], df_dx)

print(f_func(3))     # Output: 9
print(df_func(3))    # Output: 6 (gradient at x=3)

"""A more complex function"""

x = pt.scalar('x')
y = x**2
z = pt.sin(y)


dz_dx = pytensor.grad(z, x)

from pytensor.printing import debugprint
debugprint(dz_dx)

z_func = pytensor.function([x], z)
dzdx_func = pytensor.function([x], dz_dx)

print(z_func(2))       # sin(4)
print(dzdx_func(2))    # derivative using chain rule: cos(4)*2x = cos(4)*4