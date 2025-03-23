#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings

import arviz as az
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import bambi as bmb

warnings.simplefilter(action='ignore', category=FutureWarning)


# In[2]:


az.style.use("arviz-darkgrid")
# from cycler import cycler


# cmap = mpl.colormaps['gray']
# gray_cycler = cycler(color=cmap(np.linspace(0, 0.9, 6)))


# default_cycler = cycler(color=["#000000", "#6a6a6a", "#bebebe", "#2a2eec"])
# plt.rc("axes", prop_cycle=default_cycler)
plt.rc("figure", dpi=150)
np.random.seed(123)


# In[3]:


SIZE = 117
data = pd.DataFrame(
    {
        "y": np.random.normal(size=SIZE),
        "x": np.random.normal(size=SIZE),
        "z": np.random.normal(size=SIZE),
        "g": ["Group A", "Group B", "Group C"] * 39,
    }
)
data.head()


# In[4]:


a_model = bmb.Model("y ~ x", data)
a_model


# In[5]:


priors = {"x": bmb.Prior("HalfNormal", sigma=3),
          "sigma": bmb.Prior("Gamma",  mu=1, sigma=2),
          }
a_model_wcp = bmb.Model("y ~ x", data, priors=priors)
a_model_wcp


# In[6]:


no_intercept_model = bmb.Model("y ~ x", data)
no_intercept_model


# In[7]:


model_2 = bmb.Model("y ~ x + z", data)
model_2


# In[8]:


model_h = bmb.Model("y ~ x + z + (x | g)", data)
model_h


# In[9]:


model_h.build()
model_h.graph(name="../fig/bambi_dag")


# ## Simple linear regression

# In[3]:


bikes = pd.read_csv("https://github.com/aloctavodia/BAP3/raw/refs/heads/main/code/data/bikes.csv")


# In[4]:


model_t = bmb.Model("rented ~ temperature", bikes, family="negativebinomial")
idata_t = model_t.fit(random_seed=123)
model_t


# In[12]:


model_t.graph(name="../fig/bambi_linear_bikes_dag")


# In[5]:


_, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 4)) 
bmb.interpret.plot_predictions(model_t, idata_t, "temperature", ax=axes[0])
bmb.interpret.plot_predictions(model_t, idata_t, "temperature", pps=True, ax=axes[1])
axes[0].plot(bikes.temperature, bikes.rented, "C2.", zorder=-3)
axes[1].plot(bikes.temperature, bikes.rented, "C2.", zorder=-3)
axes[0].set_title("mean")
axes[1].set_title("predictions");
# plt.savefig("../fig/bambi_linear_bikes_mean_pss.png")


# In[6]:


model_th = bmb.Model("rented ~ temperature + humidity", bikes, family="negativebinomial")
idata_th = model_th.fit(random_seed=123)


# In[7]:


model_th


# In[22]:


idata_th


# In[25]:


bmb.interpret.plot_predictions(model_th, idata_th, 
                               conditional={"temperature": bikes["temperature"],
                                            "humidity": [0.18, 0.5, 0.635, 0.76, 1.0]},
                               subplot_kwargs={"group":None, "panel":"humidity"},
                               legend=False, 
                               fig_kwargs={"sharey":True, "sharex":True});
# plt.savefig("../fig/bambi_linear_bikes_th_mean.png")


# ## Polynomial

# In[26]:


model_poly_1 = bmb.Model("rented ~ hour", bikes, family="negativebinomial")
idata_poly_1 = model_poly_1.fit(random_seed=123)
model_poly_4 = bmb.Model("rented ~ poly(hour, degree=4)", bikes, family="negativebinomial")
idata_poly_4 = model_poly_4.fit(random_seed=123)


# In[27]:


_, axes = plt.subplots(2, 2, sharey=True, sharex="col", figsize=(12, 6)) 
bmb.interpret.plot_predictions(model_poly_1, idata_poly_1, "hour", ax=axes[0, 0])
bmb.interpret.plot_predictions(model_poly_1, idata_poly_1, "hour", pps=True, ax=axes[0, 1])
bmb.interpret.plot_predictions(model_poly_4, idata_poly_4, "hour", ax=axes[1, 0])
bmb.interpret.plot_predictions(model_poly_4, idata_poly_4, "hour", pps=True, ax=axes[1, 1])

for ax in axes.ravel():
    ax.plot(bikes.hour, bikes.rented, "C2.", zorder=-3)
axes[0, 0].set_title("mean")
axes[0, 1].set_title("predictions")
axes[0, 0].set_xlabel("")
axes[0, 1].set_xlabel("")
axes[0, 1].set_ylabel("")
axes[1, 1].set_ylabel("")

# plt.savefig("../fig/bambi_poly_bikes_mean_pss.png")


# ## Splines

# In[8]:


num_knots = 6
knots = np.linspace(0, 23, num_knots+2)[1:-1]
model_spline = bmb.Model("rented ~ bs(hour, degree=3, knots=knots)", bikes, family="negativebinomial")
idata_spline = model_spline.fit(random_seed=123)


# In[29]:


_, ax = plt.subplots(sharey=True, sharex="col", figsize=(12, 4)) 
bmb.interpret.plot_predictions(model_spline, idata_spline, "hour", ax=ax)
ax.plot(bikes.hour, bikes.rented, "C2.", zorder=-3)
# plt.savefig("../fig/bambi_spline_bikes.png");


# ## Distributional models

# In[9]:


babies = pd.read_csv("https://github.com/aloctavodia/BAP3/raw/refs/heads/main/code/data/babies.csv")


# In[10]:


formula = bmb.Formula(
    "length ~ np.sqrt(month)",
    "sigma ~ month"
)
model_dis = bmb.Model(formula, babies)
idata_dis = model_dis.fit(random_seed=123, include_response_params=True)


# In[11]:


model_cons = bmb.Model("length ~ np.sqrt(month)", babies)
idata_cons = model_cons.fit(random_seed=123, include_response_params=True)


# In[33]:


fig, ax = plt.subplots(figsize=(12, 4))
from matplotlib.lines import Line2D

for idx in idata_dis.posterior.coords.get("__obs__"):
    values = idata_dis.posterior["sigma"].sel(__obs__=idx).to_numpy().flatten()
    grid, pdf = az.kde(values)
    ax.plot(grid, pdf, lw=1, color="C1")

values = idata_cons.posterior["sigma"].to_numpy().flatten()
grid, pdf = az.kde(values)
ax.plot(grid, pdf, lw=3, color="C0");

# Create legend
handles = [
    Line2D([0], [0], label="Varying sigma", lw=1.5, color="k", alpha=0.6),
    Line2D([0], [0], label="Constant sigma", lw=1.5, color="C0")
]

legend = ax.legend(handles=handles, loc="upper right", fontsize=14)

ax.set(xlabel="Alpha posterior", ylabel="Density")
# plt.savefig("../fig/babies_bambi_varying.png")


# In[12]:


_, ax = plt.subplots(sharey=True, sharex="col", figsize=(12, 6)) 
bmb.interpret.plot_predictions(model_cons, idata_cons, "month", ax=ax, fig_kwargs={"color":"k"})
bmb.interpret.plot_predictions(model_cons, idata_cons, "month", pps=True, ax=ax)
ax_ = bmb.interpret.plot_predictions(model_cons, idata_cons, "month", pps=True, ax=ax, prob=0.65)
ax_[1][0].get_children()[5].set_facecolor('C1')  

ax.plot(babies.month, babies.length, "C2.", zorder=-3)


# In[34]:


_, ax = plt.subplots(sharey=True, sharex="col", figsize=(12, 6)) 
bmb.interpret.plot_predictions(model_dis, idata_dis, "month", ax=ax, fig_kwargs={"color":"k"})
bmb.interpret.plot_predictions(model_dis, idata_dis, "month", pps=True, ax=ax)
ax_ = bmb.interpret.plot_predictions(model_dis, idata_dis, "month", pps=True, ax=ax, prob=0.65)
ax_[1][0].get_children()[5].set_facecolor('C1')  

ax.plot(babies.month, babies.length, "C2.", zorder=-3)
# plt.savefig("../fig/babies_bambi_fit.png");


# In[35]:


model_dis.predict(idata_dis, kind="pps", data=pd.DataFrame({"month":[0.5]}))


# In[36]:


ref = 52.5
y_ppc = idata_dis.posterior_predictive["length"].stack(sample=("chain", "draw"))
grid, pdf = az.stats.density_utils._kde_linear(y_ppc.values)
plt.plot(grid, pdf)
percentile = int((y_ppc <= ref).mean() * 100)
plt.fill_between(
    grid[grid < ref],
    pdf[grid < ref],
    label="percentile = {:2d}".format(percentile),
    color="C2",
)
plt.xlabel("length")
plt.yticks([])
plt.legend()
# plt.savefig("../fig/babies_ppc_bambi.png")


# ## Categorical Predictors

# In[14]:


penguins = pd.read_csv("https://github.com/aloctavodia/BAP3/raw/refs/heads/main/code/data/penguins.csv").dropna()
penguins.head()


# In[15]:


# Get unique categories
unique_categories = np.unique(penguins.species)

# Create color map for categories
category_color_map = {cat: f"C{i}" for i, cat in enumerate(unique_categories)}

# Generate colors for each category
colors = [category_color_map[cat] for cat in penguins.species]

# Create scatter plot for each category
for cat in unique_categories:
    category_data = penguins[penguins.species == cat]
    plt.scatter(category_data.body_mass, category_data.bill_length, c=category_color_map[cat], label=cat)

# Add labels and legend
plt.ylabel("Body mass (g)")
plt.xlabel("Bill length (mm)")
plt.legend(labels=unique_categories, loc="lower right")
# plt.savefig("../fig/penguins_scatter.png")


# In[39]:


model_p = bmb.Model("body_mass ~ bill_length + species", data=penguins)
idata_p = model_p.fit(random_seed=123)


# In[40]:


ax = az.plot_forest(idata_p, combined=True, figsize=(12, 3));
mean_chinstrap = idata_p.posterior["species"].sel(species_dim="Chinstrap").mean()
mean_gentoo = idata_p.posterior["species"].sel(species_dim="Gentoo").mean()
ax[0].annotate(f"{mean_chinstrap.item():.2f}", (mean_chinstrap , 2.5), weight='bold')
ax[0].annotate(f"{mean_gentoo.item():.2f}", (mean_gentoo , 1.7), weight='bold')
# plt.savefig("../fig/penguins_posterior.png")


# In[41]:


bmb.interpret.plot_predictions(model_p, idata_p, ["bill_length",  "species"], fig_kwargs={"figsize":(11, 4)})
# plt.savefig("../fig/penguins_predictions.png")


# ## Interactions

# In[16]:


model_int = bmb.Model("body_mass ~ bill_depth + bill_length + bill_depth:bill_length", data=penguins)
idata_int = model_int.fit(random_seed=123)


model_noint = bmb.Model("body_mass ~ bill_depth + bill_length", data=penguins)
idata_noint = model_noint.fit(random_seed=123)


# In[17]:


plt.rc("axes")

_, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={'hspace': 0.1})

bmb.interpret.plot_predictions(model_noint, idata_noint,
                                {"bill_depth": penguins["bill_depth"],  "bill_length":[3.21, 3.95, 4.45, 4.86, 5.96]},
                                ax=axes[0],
                                legend=False
                                )
axes[0].set_title("No interaction")


bmb.interpret.plot_predictions(model_int, idata_int,
                               {"bill_depth": penguins["bill_depth"],  "bill_length":[3.21, 3.95, 4.45, 4.86, 5.96]},
                                ax=axes[1],
                                )
axes[1].set_title("Interaction")


# plt.savefig("../fig/interaction_vs_noninteraction.png")


# In[46]:


bmb.interpret.plot_comparisons(model_int, idata_int,
                               contrast={"bill_depth":[1.4, 1.8]},
                               conditional={"bill_length":[3.5, 4.5, 5.5]},
                               fig_kwargs={"figsize": (12, 3)})
# plt.savefig("../fig/interaction_comparisons.png")


# In[47]:


bmb.interpret.plot_slopes(model_int, idata_int,
                          wrt={"bill_depth":1.8},
                          conditional={"bill_length":[3.5, 4.5, 5.5]},
                          fig_kwargs={"figsize": (12, 3)})
# plt.savefig("../fig/interaction_slopes.png")


# ## Variable selection with Kulprit

# In[48]:


import kulprit as kpt


# In[49]:


body = pd.read_csv("https://github.com/aloctavodia/BAP3/raw/refs/heads/main/code/data/body_fat.csv")


# In[50]:


body.head()


# In[51]:


model = bmb.Model("siri ~ age + weight + height + abdomen + thigh + wrist", data=body)
idata = model.fit(idata_kwargs={'log_likelihood': True}, random_seed=123)


# In[52]:


ppi = kpt.ProjectionPredictive(model, idata)
ppi.search()
ppi


# In[53]:


submodel = ppi.project(3)
submodel


# In[54]:


cmp, ax = ppi.plot_compare(plot=True, figsize=(11, 4));
# plt.savefig("../fig/body_compare.png")


# In[55]:


az.plot_posterior(submodel.idata, figsize=(12, 6));
# plt.savefig("../fig/body_projected_posterior.png")


# In[56]:


"'~' ".join(submodel.term_names)


# In[57]:


ppi.plot_densities(var_names=["~Intercept", "~age", "~height", "~thigh"],
                   submodels=[3],
                   kind="forest",
                   figsize=(11, 4),
                   plot_kwargs={"colors":["C0", "C2"]});
# plt.savefig("../fig/body_true_projected.png")

