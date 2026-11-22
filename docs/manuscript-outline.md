# Meadow manuscript working outline

Working title: **Meadow estimates shared and species-specific effective migration surfaces**

Central question: **Which spatial patterns of effective migration are shared across species, and where do individual species depart from them?**

Core contribution, based on the current implementation: Meadow jointly fits SNP-derived genetic distances using graph effective resistance, with a shared edge network and species-specific deviation networks. This provides a framework for estimating a common spatial component together with individual species surfaces.

This is a manuscript plan, not a statement that all proposed results have been established. Notebook code, saved outputs, and selected saved plots were reviewed; analyses were not rerun. Saved outputs may reflect earlier code or interactive notebook states.

## Narrative and scope

The paper should establish three things:

1. The shared/species-specific decomposition recovers useful spatial structure under controlled conditions.
2. Joint inference offers a measurable benefit under some sampling and biological conditions while preserving genuine species differences.
3. An empirical example demonstrates a biologically informative common pattern and supported departures.

Keep the main empirical example undecided until candidates have been evaluated. Select for sampling coverage, model adequacy, reproducibility, and biological interpretability, rather than visual contrast alone.

For a Nature Ecology & Evolution Article, plan for approximately 500 words of Introduction, 2,000 words of Results, and 1,000 words of Discussion. The current guidelines permit 3,500 words of main text, a 200-word abstract, and six main display items; Methods are outside the main-text word count. Source: https://www.nature.com/natecolevol/content

## Introduction: a comparative view of landscape connectivity

Paragraph 1 — Biological question. Explain why understanding agreement and differences in spatial genetic structure across species matters for comparative landscape genetics.

Paragraph 2 — Existing inference. Introduce single-species effective migration and resistance approaches. Distinguish learning spatial genetic structure from directly measuring contemporary movement.

Paragraph 3 — Gap. Motivate estimating a common spatial component and species departures jointly. Explain the competing risks of fitting each species separately and forcing all species to share the same surface. Position relative to relevant comparative landscape-genetic methods after a focused literature review; avoid an unsupported first-method claim.

Paragraph 4 — Contribution. Introduce Meadow, its outputs, and the validation and empirical findings actually supported by the completed analyses.

## Results 1: a joint model of shared and species-specific resistance

Explain the model intuitively: samples and genotypes -> spatial graph and edge features -> shared and species-specific edge scores -> positive edge resistance -> graph effective resistance -> predicted genetic distances.

The current implementation adds shared and species-specific scores before a softplus transformation. Resistance values themselves are therefore not an additive shared-plus-deviation decomposition. Describe departures on the model score scale or define a clear resistance-scale contrast to the shared reference.

Define what the shared component estimates, how it is normalized, and how regularization determines the allocation of structure between components. It is not automatically an arithmetic average of species surfaces or a feature present equally in every species. Explain species-specific distance calibration and the interpretation of relative surface scale.

**Figure 1:** Conceptual model with a common feature and a species-specific departure; show the shared reference, signed deviations, and full species surfaces as distinct outputs.

## Results 2: recovering shared structure and species differences in simulations

Build on `examples/Simulations/Meadow-Simulation-Test.ipynb`, which contains:

- A strong common central barrier across four simulated species.
- Mixed horizontal and vertical barrier orientations.
- A common barrier with patchy sampling (`sample_prob=0.25`).
- Uniform migration, with a notebook note about edge effects.

Add a scenario with an explicitly shared feature plus an additional species-specific feature. Quantify recovery separately for the shared component and the full species surfaces, using appropriate normalization. Repeat scenarios across random seeds and report variability.

The mixed-orientation scenario is useful for testing conflicting information. It should not be treated as having a uniquely defined common truth without specifying the estimand. Treat uniform migration as an essential null test; the saved plot shows spatial variation that needs to be understood before claiming reliable feature recovery.

**Figure 2:** Simulation truth and inferred patterns for representative scenarios, plus quantitative recovery and null-case results across replicates.

## Results 3: when sharing information improves inference

Compare Meadow with independently fitted species models, a shared-only model, and a clearly defined aggregation of independently estimated surfaces. Use established single-species approaches where appropriate and feasible.

Vary species number, spatial overlap, sampling density, information per species, and strength of species-specific departures. Measure held-out genetic-distance prediction as well as surface recovery; one does not establish the other.

Test whether benefits persist when species disagree. Include sensitivity to the shared/species regularization balance, mesh resolution, environmental features versus coordinate features, and graph support attenuation.

Use explicit spatial holdouts for claims about spatial prediction. The default training split can fall back from site holdout to random pair holdout, so saved validation losses should not automatically be described as spatial cross-validation.

**Figure 3:** Joint-versus-independent performance across controlled conditions, including cases with little shared structure.

## Results 4: shared and species-specific structure in an empirical landscape

**Empirical system remains open.** Begin with the sampling design and a biological question. Present the shared component, selected full species surfaces, and signed departures. Assess whether the shared feature persists when individual species are omitted and when analysis choices change.

Potential candidate roles based on repository inspection:

| Candidate | Evidence currently present | Possible role / key limitation |
| --- | --- | --- |
| PEDIC / Hengduan workflow | `notebooks/pedic_example.ipynb` loads 12 species and contains fitted maps and a spatial-CV summary; data and graph paths point outside this repository | Existing application to develop further; inspect geographic support and stability before assigning a biological interpretation |
| Eastern snakes | Nine genotype/coordinate datasets; 9–144 individuals and 9–133 distinct coordinate pairs per species; fitted maps and a smoothing-CV summary | Existing multispecies candidate; uneven coverage and graph support need explicit assessment |
| Mississippi fish | Six species, 179–256 individuals per species, but only six distinct coordinates per species; mesh/boundary experiments and smoothing CV | Useful sparse-spatial-sampling example; limited information for resolving a detailed surface |
| Panama riverweed | One focal species split by drainage and nuclear/plastome data; roughly 3–4 locations per partition in the current coordinate files | Exploratory extension; does not directly demonstrate the central multispecies contribution |
| Rossetto rainforest trees | Summary table lists 22 species, 17–58 sites and 80–290 individuals per species; filtered genotype objects, coordinates in summary tables, and original analysis scripts | Strong candidate for the next suitability assessment; no Meadow analysis found, and genotype/coordinate extraction and coverage checks remain |
| Angiosperm community workflow | A community presence/absence matrix represented as one `SpeciesData` object, compared with population-genetic outputs | Possible later extension; requires a separate interpretation and validation from multispecies SNP inference |

**Figure 4:** Empirical sampling coverage, shared reference surface, and robustness/support summaries.

**Figure 5:** Selected full species surfaces and signed departures, with quantitative evidence for agreement and differences.

A second empirical application is optional. Include it in the main text only if it adds a distinct result about generality; otherwise use Extended Data.

## Discussion

1. State the supported empirical and methodological findings.
2. Explain what the shared/deviation representation adds to comparative inference.
3. Discuss the relationship between common genetic structure and possible landscape or historical explanations, without implying causal attribution from map overlap alone.
4. Address effective migration versus actual movement, regularization-dependent decomposition, uneven geographic support, graph geometry, and uncertainty/stability.
5. Describe the comparative questions Meadow enables and the data designs best suited to them.

## Methods

- Sample-level inputs, genotype processing, missingness, aggregation, and genetic-distance targets.
- Shared graph construction, sample assignment, environmental and coordinate edge features.
- Shared and species-specific neural edge models and positivity transformation.
- Effective resistance calculation and species-specific distance calibration.
- Objective function, regularization, smoothing, identifiability/normalization, and optimization.
- Spatial validation design, hyperparameter selection, random seeds, and repeated-fit stability.
- Simulation design, comparison models, recovery metrics, and prediction metrics.
- Empirical provenance, filtering, graph choices, geographic support, and sensitivity analyses.
- Software, code/data availability, and computational requirements.

## Priorities before filling in result claims

1. Define the shared estimand and deviation scale; evaluate sensitivity to regularization and initialization.
2. Quantify the existing simulation scenarios, including the uniform-migration null and graph boundary effects.
3. Add an explicit shared-plus-species-specific simulation and independent/shared-only baselines.
4. Assess rainforest-tree input compatibility and geographic overlap; compare suitability with PEDIC and snakes.
5. Freeze reproducible empirical workflows and choose the main application using both predictive/model checks and biological interpretability.
6. Write result headings as concrete findings only after these checks establish them.
