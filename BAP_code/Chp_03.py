#!/usr/bin/env python
# coding: utf-8

# In[1]:


import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import preliz as pz


# In[2]:


# az.style.use("arviz-grayscale")
# from cycler import cycler
# default_cycler = cycler(color=["#000000", "#6a6a6a", "#bebebe", "#2a2eec"])
# plt.rc('axes', prop_cycle=default_cycler)
# plt.rc('figure', dpi=300)
np.random.seed(123)


# In[3]:


cs_data = pd.read_csv('data/chemical_shifts_theo_exp.csv')
diff = cs_data.theo - cs_data.exp
cat_encode = pd.Categorical(cs_data['aa'])
idx = cat_encode.codes
coords = {"aa": cat_encode.categories}


# In[4]:


with pm.Model(coords=coords) as cs_nh:         
    μ = pm.Normal('μ', mu=0, sigma=10, dims="aa") 
    σ = pm.HalfNormal('σ', sigma=10, dims="aa") 
 
    y = pm.Normal('y', mu=μ[idx], sigma=σ[idx], observed=diff) 
     
    idata_cs_nh = pm.sample(random_seed=4591)


# In[5]:


with pm.Model(coords=coords) as cs_h:
    # hyper_priors
    μ_mu = pm.Normal('μ_mu', mu=0, sigma=10)
    μ_sd = pm.HalfNormal('μ_sd', 10)

    # priors
    μ = pm.Normal('μ', mu=μ_mu, sigma=μ_sd, dims="aa") 
    σ = pm.HalfNormal('σ', sigma=10, dims="aa") 

    y = pm.Normal('y', mu=μ[idx], sigma=σ[idx], observed=diff) 

    idata_cs_h = pm.sample(random_seed=4591)


# In[6]:


axes = az.plot_forest([idata_cs_nh, idata_cs_h], model_names=['non_hierarchical', 'hierarchical'],
                      var_names='μ', combined=True, r_hat=False, ess=False, figsize=(10, 7),
                      colors='cycle')
y_lims = axes[0].get_ylim()
axes[0].vlines(idata_cs_h.posterior['μ_mu'].mean(), *y_lims, color="k", ls=":");
plt.savefig("../fig/csh_vs_csnh.png")


# In[7]:


for name, model in zip(("cs_nh", "cs_h"), (cs_nh, cs_h)):
    graph = pm.model_to_graphviz(model)
    graph.graph_attr.update(size="2,2!")
    graph.graph_attr.update(dpi="300")
    graph.render(filename=name, format="png", cleanup=True)


# In[8]:


N_samples = [30, 30, 30]
G_samples = [18, 18, 18]
group_idx = np.repeat(np.arange(len(N_samples)), N_samples)
data = []
for i in range(0, len(N_samples)):
    data.extend(np.repeat([1, 0], [G_samples[i], N_samples[i]-G_samples[i]]))


# In[9]:


with pm.Model() as model_h:
    # hypyerpriors
    μ = pm.Beta('μ', 1., 1.)
    ν = pm.HalfNormal('ν', 10)
    # prior
    θ = pm.Beta('θ', mu=μ, nu=ν, shape=len(N_samples))
    # likelihood
    y = pm.Bernoulli('y', p=θ[group_idx], observed=data)

    idata_h = pm.sample(random_seed=4591)
    


# In[10]:


az.plot_trace(idata_h)
plt.savefig("../fig/idata_h_trace.png")


# In[11]:


posterior = az.extract(idata_h, num_samples=100)
for sample in posterior[["μ", "ν"]].to_array().values.T:
    pz.Beta(mu=sample[0], nu=sample[1]).plot_pdf(legend=None, color="C0", alpha=0.1, support=(0.01, 0.99), moments="m")

pz.Beta(mu=posterior["μ"].mean().item(), nu=posterior["ν"].mean().item()).plot_pdf(legend=None, color="C0", moments="m")
plt.xlabel('$θ_{prior}$')
plt.savefig("../fig/idata_h_posterior.png")


# In[12]:


for sample in posterior[["μ", "ν"]].to_array().values.T:
    print(sample)


# In[13]:


football = pd.read_csv("data/football_players.csv", dtype={'position':'category'})
football


# In[14]:


# Define the constraints
lower_bound = 0.0
upper_bound = 0.5
desired_mass = 0.95

dist = pz.Beta()
pz.maxent(dist, lower=lower_bound, upper=upper_bound, mass=desired_mass)
print(f"α = {dist.alpha:.2f}, β = {dist.beta:.2f}")


# In[15]:


lower_bound = 50
upper_bound = 200
desired_mass = 0.90

dist = pz.Gamma()
pz.maxent(dist, lower=lower_bound, upper=upper_bound, mass=desired_mass)
print(f"mu = {dist.mu:.2f}, sigma = {dist.sigma:.2f}")


# In[16]:


def try_model_mu_p(nu_mu_pos=50, nu_sigma_pos=60, mu_alpha_pos=1.7, mu_beta_pos=5.8):
    # Hyper parameters
    μ_pos = pz.Beta(mu_alpha_pos, mu_beta_pos).rvs()
    ν_pos = pz.Gamma(mu=nu_mu_pos, sigma=nu_sigma_pos).rvs()

    # Parameters for positions
    μ_p = pz.Beta(mu=μ_pos, nu=ν_pos).rvs(100)

    return μ_p


# In[ ]:


pz.predictive_explorer(try_model_mu_p)


# In[18]:


def try_model_theta(nu_mu_pos=50, nu_sigma_pos=60, nu_mu=150, nu_sigma=200, mu_alpha_pos=1.7, mu_beta_pos=5.8):
    # Hyper parameters
    μ_pos = pz.Beta(mu_alpha_pos, mu_beta_pos).rvs()
    ν_pos = pz.Gamma(mu=nu_mu_pos, sigma=nu_sigma_pos).rvs()

    # Parameters for positions
    μ_p = pz.Beta(mu=μ_pos, nu=ν_pos).rvs()
    
    ν_p = pz.Gamma(mu=nu_mu, sigma=nu_sigma).rvs()

    # Parameter for players
    θ = pz.Beta(mu=μ_p, nu=ν_p).rvs(100)

    return θ


# In[93]:


pz.predictive_explorer(try_model_theta)


# In[19]:


pos_idx = football.position.cat.codes.values
pos_codes = football.position.cat.categories
n_pos = pos_codes.size
n_players = football.index.size


# In[ ]:


coords = {"pos": pos_codes}
with pm.Model(coords=coords) as model_football:
    # Hyper parameters
    μ = pm.Beta('μ', 1.7, 5.8) 
    ν = pm.Gamma('ν', mu=125, sigma=50)

    
    # Parameters for positions
    μ_p = pm.Beta('μ_p',
                       mu=μ,
                       nu=ν,
                       dims = "pos")
    
    ν_p = pm.Gamma('ν_p', mu=125, sigma=50, dims="pos")
 
    # Parameter for players
    θ = pm.Beta('θ', 
                    mu=μ_p[pos_idx],
                    nu=ν_p[pos_idx])
    
    _ = pm.Binomial('gs', n=football.shots.values, p=θ, observed=football.goals.values)

    idata_football = pm.sample(draws=3000, target_accept=0.95, random_seed=4591)


# In[94]:


graph = pm.model_to_graphviz(model_football)
graph.graph_attr.update(size="4,4!")
graph.graph_attr.update(dpi="300")
graph.render(filename="beta_binomial_hierarchical_subjects_dag", format="png", cleanup=True)


# In[96]:


_, ax = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
az.plot_posterior(idata_football, var_names='μ', ax=ax[0])
ax[0].set_title(r"Global mean")
az.plot_posterior(idata_football.posterior.sel(pos="FW"), var_names='μ_p', ax=ax[1])
ax[1].set_title(r"Forward position mean")
az.plot_posterior(idata_football.posterior.sel(θ_dim_0=1457), var_names='θ', ax=ax[2])
ax[2].set_title(r"Messi mean")
plt.savefig("../fig/beta_binomial_hierarchical_subjects_global_mus.png")


# In[18]:


az.plot_forest(idata_football, var_names=['μ_p'], combined=True, figsize=(12, 3))
plt.savefig("../fig/beta_binomial_hierarchical_subjects_positions.png")


# In[99]:


_, ax = plt.subplots(1, 1, figsize=(12, 6), sharex=True)
az.plot_posterior(idata_football, var_names='μ', ax=ax)
ax.set_title(r"Global mean")


# In[ ]:


fig, axes = plt.subplots(4, 1, figsize=(19, 14), sharex=True)
az.plot_posterior(idata_football, var_names='μ_p', grid=[4,1], ax=axes)
plt.tight_layout()


# In[125]:


theta = idata_football.posterior['θ']
theta = theta.assign_coords(pos=('θ_dim_0', pos_codes[pos_idx]))
theta_forward = theta.sel(θ_dim_0=theta.pos == "FW")
theta_forward


# In[130]:


idata_football.posterior['μ_p'].pos


# In[ ]:


sorted_indices = np.argsort(theta_values)

fig, axes = plt.subplots(2, 1, figsize=(19, 14), sharex=True)
az.plot_posterior(idata_football, var_names='μ_p', coords={'pos':'FW'}, ax=axes[0])
az.plot_forest(theta_forward, ax=axes[1], 
               coords={'θ_dim_0': theta_forward.coords['θ_dim_0'][sorted_indexes]}, c
               ombined=True)


# In[136]:


theta_goalie = theta.sel(θ_dim_0=theta.pos == "GK")

fig, axes = plt.subplots(2, 1, figsize=(19, 14), sharex=True)
az.plot_posterior(idata_football, var_names='μ_p', coords={'pos':'GK'}, ax=axes[0])
az.plot_forest(theta_goalie, ax=axes[1], 
               # coords={'θ_dim_0': theta_forward.coords['θ_dim_0'][0:100]}, 
               combined=True)


# In[143]:


fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
az.plot_posterior(theta_goalie.std(dim='θ_dim_0'), ax=axes[0])
axes[0].set_title("Standard deviation for goalkeepers")
az.plot_posterior(theta_forward.std(dim='θ_dim_0'), ax=axes[1])
axes[1].set_title("Standard deviation for forwards")


# In[137]:


theta_goalie

