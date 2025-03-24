### Lecture 5A-B: MCMC - PowerPoint Draft Content

---

#### 5A Review: Bayesian Workflow Recap (5 slides)

1. **Slide Title:** Bayesian Workflow Overview
   - Bullet Points:
     - Bayesian workflow guides data analysis from start to finish. [A1]
     - Each step influences modeling decisions and inference. [A2]
   - Figure: Lecture 4 Diagnostics.pptx, Slide 5 (Workflow diagram)  
     - Appears immediately, no animation.

2. **Slide Title:** Components of the Bayesian Workflow
   - Bullet Points:
     - Data → Model → Prior → Sampling → Diagnostics → Posterior Predictive → Inference. [A1]
     - Each component has specific tools and checks. [A2]
   - Figure: Lecture 4 Diagnostics.pptx, Slide 6 (Expanded workflow)  
     - Appears immediately, no animation.

3. **Slide Title:** Focus on Sampling Step
   - Bullet Points:
     - Today’s focus: Sampling step. [A1]
     - Sampling needed when models too complex for analytic solutions. [A2]
   - Figure: Lecture 4 Diagnostics.pptx, Slide 6 (Sampling highlighted)  
     - Sampling highlight appears at [A1].

4. **Slide Title:** Recap: Normal Models & Sampling
   - Bullet Points:
     - Last time: Simple Normal models, priors, and sampling basics. [A1]
     - Next step: More complex models need new tools. [A2]
   - Figure: Lecture 4 Diagnostics.pptx, Slides 17-19 (Gaussian model diagrams)  
     - Appears immediately, no animation.

5. **Slide Title:** Today’s Focus: MCMC
   - Bullet Points:
     - Goal: Understand and apply MCMC methods. [A1]
     - MCMC allows scalable sampling in complex models. [A2]
   - Textbox:
     - Sampling is the heart of modern Bayesian analysis. MCMC makes it practical. [A1]
   - Figure: New Figure Needed (Highlight 'Sample' step) [A1]

---

#### 5B Why Sampling? (4 slides)

1. **Slide Title:** Why Sampling? Analytical Challenges
   - Bullet Points:
     - Analytic solutions require conjugate priors. [A1]
     - Rarely applicable for complex models. [A2]
   - Figure: No figure.

2. **Slide Title:** Analytical Solutions: Limitations
   - Bullet Points:
     - Conjugate priors only available for special cases. [A1]
     - Real data models are rarely so simple. [A2]
   - Figure: No figure.

3. **Slide Title:** Complex Models: Too Complicated for Closed Form
   - Bullet Points:
     - High-dimensional, hierarchical, or non-linear models. [A1]
     - Require alternative solutions beyond closed form. [A2]
   - Figure: New Figure Needed (Example of complex model schematic). Appears immediately.

4. **Slide Title:** Transition to Grid Approximation
   - Bullet Points:
     - Grid approximation was an early solution attempt. [A1]
     - But fails badly in higher dimensions. [A2]
   - Figure: Lecture 3 MCMC.pptx, Slide 7 (Grid computing intro). Appears immediately.

---

#### 5C The Grid Method Recap

1. **Title Slide:** Grid Method Recap
   - No figure, title slide only.

2. **Slide Title:** What is Grid Approximation?
   - Bullet Points:
     - Grid method approximates posterior by discretizing parameter space. [A1]
     - Feasible for simple, low-dimensional models. [A2]
   - Figure: Lecture 3 MCMC.pptx, Slide 8 (BAP Fig 8.1 Grid plot). Appears immediately.

3. **Slide Title:** Coin Flip Grid Posterior Example
   - Bullet Points:
     - Example: Coin flips modeled with grid approximation. [A1]
     - Discrete grid points evaluated. [A2]
   - Figure: Generated using Lecture5_Figures notebook (grid_coin_flip.png). Appears immediately.

4. **Slide Title:** Limitations: Curse of Dimensionality
   - Bullet Points:
     - Grid size grows exponentially with dimensions. [A1]
     - Quickly becomes impractical. [A2]
   - Figure: New Figure Needed (dimension scaling schematic). Appears immediately.

5. **Slide Title:** Computational Cost of Grid Methods
   - Bullet Points:
     - Requires evaluating entire parameter grid. [A1]
     - Infeasible as complexity increases. [A2]
   - Figure: New Figure Needed (computational time vs dimension plot). Appears immediately.

6. **Slide Title:** Why MCMC?
   - Bullet Points:
     - MCMC offers scalable sampling even in complex spaces. [A1]
     - No need to discretize entire parameter space. [A2]
   - Figure: No figure.

---

#### 5D Enter MCMC

1. **Title Slide:** Enter MCMC
   - No figure, title slide only.

2. **Slide Title:** Core Idea: Markov Chain + Monte Carlo
   - Bullet Points:
     - Combine Markov Chain's dependence on previous state. [A1]
     - Monte Carlo uses randomness to approximate distributions. [A2]
   - Figure: New Figure Needed (Chain path visualization). Appears immediately.

3. **Slide Title:** Markov Chain Property
   - Bullet Points:
     - Next sample depends only on current state. [A1]
     - Memoryless property key to MCMC. [A2]
   - Figure: New Figure Needed (Markov Chain diagram). Appears immediately.

4. **Slide Title:** Monte Carlo Sampling
   - Bullet Points:
     - Approximates distributions by random sampling. [A1]
     - Becomes more accurate with more samples. [A2]
   - Figure: New Figure Needed (Random samples plot). Appears immediately.

5. **Slide Title:** Why MCMC Solves Grid Problems
   - Bullet Points:
     - No need to evaluate all grid points. [A1]
     - Samples focus on high-probability regions. [A2]
   - Figure: No figure.

---

#### 5E Metropolis Algorithm

1. **Title Slide:** Metropolis Algorithm
   - No figure, title slide only.

2. **Slide Title:** Historical Context
   - Bullet Points:
     - Developed in the 1950s for physics simulations. [A1]
     - Basis for many MCMC methods. [A2]
   - Figure: New Figure Needed (Metropolis historical illustration). Appears immediately.

3. **Slide Title:** Algorithm Overview
   - Bullet Points:
     - Propose new value based on current state. [A1]
     - Accept/reject based on acceptance ratio. [A2]
   - Figure: Lecture 3 MCMC.pptx, Slide 18 (Algorithm breakdown). Appears immediately.

4. **Slide Title:** Step 1: Propose New Value
   - Bullet Points:
     - Proposals drawn from a simple distribution. [A1]
     - Commonly Normal or Uniform proposals. [A2]
   - Figure: Lecture 3 MCMC.pptx, Slide 18. Appears immediately.

5. **Slide Title:** Step 2: Calculate Acceptance Ratio
   - Bullet Points:
     - Ratio compares new and old posterior probabilities. [A1]
     - Accept based on likelihood ratio and prior. [A2]
   - Figure: Lecture 3 MCMC.pptx, Slide 18. Appears immediately.

6. **Slide Title:** Step 3: Accept/Reject Rule
   - Bullet Points:
     - Accept if acceptance ratio > random threshold. [A1]
     - Otherwise, stay at current state. [A2]
   - Figure: Lecture 3 MCMC.pptx, Slide 18. Appears immediately.

7. **Slide Title:** Visualizing Acceptance
   - Bullet Points:
     - Trace shows samples moving across state space. [A1]
     - Can visualize random walk behavior. [A2]
   - Figure: Lecture 3 MCMC.pptx, Slide 19 (Random walk example). Appears immediately.

8. **Slide Title:** Beta(2,5) Sampling Example
   - Bullet Points:
     - Simple Metropolis applied to Beta(2,5). [A1]
     - Visualizes convergence to target distribution. [A2]
   - Figure: Generated from Lecture5_Figures notebook (metropolis_beta.png). Appears immediately.

9. **Slide Title:** Trace Plot: Convergence Behavior
   - Bullet Points:
     - Shows sample values over iterations. [A1]
     - Should stabilize as chain converges. [A2]
   - Figure: Generated from Lecture5_Figures notebook (metropolis_trace.png). Appears immediately.

10. **Slide Title:** Sample Histogram vs. True Distribution
    - Bullet Points:
      - Histogram approximates target distribution. [A1]
      - Compare to known Beta(2,5) distribution. [A2]
    - Figure: Same as Beta(2,5) figure.

11. **Slide Title:** Proposal Distribution Tuning
    - Bullet Points:
      - Proposal width affects efficiency. [A1]
      - Poor tuning leads to slow mixing. [A2]
    - Figure: Lecture 3 MCMC.pptx, Slide 28 (Proposal tuning visual). Appears immediately.

12. **Slide Title:** Issues: Random Walk Behavior
    - Bullet Points:
      - Basic Metropolis performs poorly in high dimensions. [A1]
      - Solutions: More advanced samplers. [A2]
    - Figure: Lecture 3 MCMC.pptx, Slide 28. Appears immediately.

---

#### 5F Diagnosing Chains

1. **Title Slide:** Diagnosing Chains
   - No figure, title slide only.

2. **Slide Title:** Why Diagnostics Matter
   - Bullet Points:
     - Diagnosing sampling quality is critical. [A1]
     - Prevents incorrect inferences. [A2]
   - Figure: No figure.

3. **Slide Title:** Diagnostic Categories
   - Bullet Points:
     - Mixing diagnostics: Trace, R-hat, Rank plots. [A1]
     - Efficiency diagnostics: ESS, Autocorrelation. [A2]
     - Divergences diagnostic: Only in HMC. [A3]
   - Figure: Lecture 4 Diagnostics.pptx, Slide 8 (Diagnostics overview). Appears immediately.

4. **Slide Title:** Trace Plots: Good vs Bad
   - Bullet Points:
     - Good: Chains overlap, stable. [A1]
     - Bad: Poor mixing, stuck chains. [A2]
   - Figure: Lecture 4 Diagnostics.pptx, Slides 13-14 (Good/Bad trace plots). Appears immediately.

5. **Slide Title:** R-hat: Convergence Statistic
   - Bullet Points:
     - Measures between-chain and within-chain variance. [A1]
     - Close to 1 → converged. [A2]
   - Figure: Lecture 4 Diagnostics.pptx, Slide 15 (R-hat equation slide). Appears immediately.

6. **Slide Title:** R-hat: Good vs Poor Examples
   - Bullet Points:
     - Good: R-hat near 1. [A1]
     - Bad: Divergence across chains. [A2]
   - Figure: Lecture 4 Diagnostics.pptx, Slide 16 (Combined R-hat table). Appears immediately.

7. **Slide Title:** Rank Plots: Chain Overlap
   - Bullet Points:
     - Checks uniformity across chains. [A1]
     - Misalignment signals convergence issues. [A2]
   - Figure: Lecture 4 Diagnostics.pptx, Slide 17 (Rank plot). Appears immediately.

8. **Slide Title:** Effective Sample Size (ESS)
   - Bullet Points:
     - Measures sample efficiency. [A1]
     - High autocorrelation → low ESS. [A2]
   - Figure: Lecture 4 Diagnostics.pptx, Slide 20 (ESS equation). Appears immediately.

9. **Slide Title:** ESS Visualization: Good vs Bad
   - Bullet Points:
     - Compare good and bad chain ESS plots. [A1]
   - Figure: Lecture 4 Diagnostics.pptx, Slides 21-22 (ESS plots). Appears immediately.

10. **Slide Title:** Recommended Diagnostic Trio
    - Bullet Points:
      - Trace plot, ESS, Divergences as default checks. [A1]
    - Figure: No figure.

11. **Slide Title:** Bad Sampling Case Study: Data & Model
    - Bullet Points:
      - Data generated from bimodal Gaussian. [A1]
      - Naive Gaussian model applied. [A2]
    - Figure: New Figure Needed (mixture Gaussians data + naive fit). Appears immediately.

12. **Slide Title:** Diagnostics Results: Bad Sampling
    - Bullet Points:
      - Trace plot stuck in one mode. [A1]
      - R-hat poor, ESS low. [A2]
    - Figure: New Figure Needed (trace, ESS, R-hat from bad example). Appears immediately.

---

#### 5G Good Practices for Sampling

1. **Title Slide:** Good Practices for Sampling
   - No figure, title slide only.

2. **Slide Title:** Prior Predictive Checks
   - Bullet Points:
     - Always check prior predictive before sampling. [A1]
     - Sanity check for model assumptions. [A2]
   - Figure: Lecture 4 Diagnostics.pptx, Slide 9 (Prior predictive). Appears immediately.

3. **Slide Title:** Choosing Reasonable Priors
   - Bullet Points:
     - Informative but flexible. [A1]
     - Avoid overly tight priors. [A2]
   - Figure: New Figure Needed (prior predictive mismatch example). Appears immediately.

4. **Slide Title:** Diagnostics Review after Sampling
   - Bullet Points:
     - R-hat close to 1, high ESS. [A1]
     - No/few divergences. [A2]
   - Figure: Reuse Lecture 4 slides (R-hat, ESS).

5. **Slide Title:** Posterior Predictive Checks
   - Bullet Points:
     - Compare model predictions to observed data. [A1]
   - Figure: Lecture 4 Diagnostics.pptx, Slide 23 (Posterior predictive). Appears immediately.

6. **Slide Title:** Practical Checklist
   - Bullet Points:
     - Prior predictive → Sampling → Diagnostics → Posterior predictive. [A1]
   - Figure: No figure.

---

#### 5H Bayesian Workflow Revisited

1. **Title Slide:** Bayesian Workflow Revisited
   - No figure, title slide only.

2. **Slide Title:** Full Workflow Recap
   - Bullet Points:
     - Revisit Data → Model → Prior → Sampling → Diagnostics → Posterior Predictive → Inference. [A1]
   - Figure: Lecture 4 Diagnostics.pptx, Slide 6 (Workflow diagram revisited). Appears immediately.

3. **Slide Title:** Emphasizing Sampling Step
   - Bullet Points:
     - Sampling is the engine of Bayesian analysis. [A1]
   - Figure: Same diagram, highlight Sampling. Appears immediately.

4. **Slide Title:** Where Diagnostics Fit
   - Bullet Points:
     - Diagnostics crucial after sampling. [A1]
   - Figure: Same diagram, highlight Diagnostics. Appears immediately.

5. **Slide Title:** Posterior Predictive & Inference
   - Bullet Points:
     - Posterior predictive leads to final inference. [A1]
   - Figure: Lecture 4 Diagnostics.pptx, Slide 23. Appears immediately.

---

#### 5I Modern MCMC: Hamiltonian Monte Carlo (HMC)

1. **Title Slide:** Hamiltonian Monte Carlo (HMC)
   - No figure, title slide only.

2. **Slide Title:** Random Walk Problem in Metropolis
   - Bullet Points:
     - Metropolis has inefficient exploration. [A1]
   - Figure: Lecture 3 MCMC.pptx, Slide 28. Appears immediately.

3. **Slide Title:** Adding Momentum: Concept of HMC
   - Bullet Points:
     - Momentum variables guide sampling. [A1]
   - Figure: Lecture 3 MCMC.pptx, Slide 34 (satellite analogy). Appears immediately.

4. **Slide Title:** Physics Analogy: Energy Decomposition
   - Bullet Points:
     - Potential and kinetic energy used in sampling. [A1]
   - Figure: Lecture 3 MCMC.pptx, Slide 35. Appears immediately.

5. **Slide Title:** Gradient Informed Proposals
   - Bullet Points:
     - Gradients guide efficient steps. [A1]
   - Figure: New Figure Needed (HMC gradient step schematic). Appears immediately.

6. **Slide Title:** Leapfrog Integrator
   - Bullet Points:
     - Ensures stability in proposals. [A1]
   - Figure: Lecture 3 MCMC.pptx, Slide 36 (Leapfrog visual). Appears immediately.

7. **Slide Title:** Efficient Long Trajectories
   - Bullet Points:
     - Trajectories explore parameter space efficiently. [A1]
   - Figure: New Figure Needed (HMC trajectory plot). Appears immediately.

8. **Slide Title:** Tuning Step Size and Trajectory Length
   - Bullet Points:
     - Poor tuning → inefficiency/divergences. [A1]
   - Figure: New Figure Needed (Step size tuning visual). Appears immediately.

9. **Slide Title:** PyMC Default Sampler
   - Bullet Points:
     - PyMC uses HMC with adaptive tuning. [A1]
   - Figure: No figure.

10. **Slide Title:** Transition: Diagnosing HMC Behavior
    - Bullet Points:
      - Diagnostics crucial after HMC sampling. [A1]
    - Figure: No figure.

---

#### 5J Diagnosing HMC

1. **Title Slide:** Diagnosing HMC
   - No figure, title slide only.

2. **Slide Title:** Divergences Overview
   - Bullet Points:
     - Divergences indicate exploration issues. [A1]
   - Figure: Lecture 4 Diagnostics.pptx, Slide 12 (Divergences plot). Appears immediately.

3. **Slide Title:** Causes of Divergences
   - Bullet Points:
     - Sharp posterior geometry. [A1]
   - Figure: New Figure Needed (Geometric divergence illustration). Appears immediately.

4. **Slide Title:** Visualizing Divergences: Example
   - Bullet Points:
     - Example: Gaussian model divergences. [A1]
   - Figure: Lecture 4 Diagnostics.pptx, Slide 12. Appears immediately.

5. **Slide Title:** Revisiting Trace & ESS
   - Bullet Points:
     - Diagnostics remain essential. [A1]
   - Figure: Lecture 4 Diagnostics.pptx, Slides 13-22 reused.

6. **Slide Title:** Summary: Diagnostic Checklist
   - Bullet Points:
     - Always check: Trace, ESS, Divergences. [A1]
   - Figure: No figure.

