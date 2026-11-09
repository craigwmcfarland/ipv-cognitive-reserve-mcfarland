# ipv-cognitive-reserve-mcfarland
Hello, Researcher! This repository hosts code to reproduce the analyses and figures in "Cognitive reserve depends on cortical neurochemistry to predict memory after partner-inflicted brain injury"

This repository includes scripts to reproduce the analyses reported in:

McFarland, C. W., Lin, A. P., McComas, S., Ratai, E.-M., Gönenç, A., Joseph Denk, A.-L., Nikolaidou, S., & Valera, E. M. (2026). Cognitive reserve depends on cortical neurochemistry to predict memory after partner-inflicted brain injury. bioRxiv. https://doi.org/10.64898/2026.10.27.762370

It contains the complete pipeline for analyzing brain injury interview, neuropsychological, computerized cognitive, questionnaire, and proton magnetic resonance spectroscopy (MRS) data from a community cohort of women with lifetime exposure to physical intimate partner violence (IPV). The data were collected as part of a study of IPV-related brain injury in the Valera Lab at Massachusetts General Hospital and Harvard Medical School (PI: Eve M. Valera). The workflow covers the following:

- Injury exposure and cohort characteristics.
- Memory outcomes.
- Whether cognitive reserve predicts memory conditional on injury burden.
- Whether cognitive reserve predicts memory conditional on cortical neurochemistry.
- Covariates and performance validity.
- The figures.

Workflow Overview:

1. Injury Exposure and Memory Outcomes (Python 3.12)

01_injury_exposure_and_cohort.py
Summarizes the distribution of cumulative partner-inflicted brain injury count, applies winsorizing at 25 and the log transform, defines the exposure bands (0, 1–9, ≥10), and reproduces Table 1.

02_memory_outcomes.py
Describes memory performance relative to published norms (Supplementary Table 1). Tests the association between injury burden and delayed recall. Reports the variance in each memory measure explained by injury burden and by cognitive reserve, for the whole cohort and within each exposure band (Supplementary Table 2).

2. Cognitive Reserve Analyses (Python 3.12)

These scripts reproduce the remaining results and supplementary tables.

03_reserve_by_injury.py

Moderator: Injury burden
Analyses:
- Reserve–memory associations within exposure bands.
- The reserve-by-injury interaction in the three-strata and binary models.
- Robustness: leave-one-out, Freedman–Lane permutation and leave-one-out cross-validation.
- Sensitivity analyses (Supplementary Table 3).
- Exposure thresholds (Supplementary Table 4).
- Injury mechanism (Supplementary Table 5).
- Specificity across CNS Vital Signs domains, including the domain-family mixed model (Supplementary Table 6).
- Alternative exposure codings (Fig. S1).

04_reserve_by_neurochemistry.py

Moderator: Creatine-referenced metabolites (tNAA/tCr, Glx/tCr, mI/tCr, Cho/tCr) in the posterior cingulate gyrus, dorsolateral prefrontal cortex and corona radiata
Analyses:
- Reserve-by-metabolite interactions with Benjamini–Hochberg correction within families (Supplementary Table 7).
- Conditional slopes.
- Robustness: BCa bootstrap, leave-one-out and cross-validation, plus sensitivity analyses (Supplementary Table 8).
- Joint moderation with injury burden, and the tNAA and tCr components.
- Other memory measures.
- Mediation.
- Spectroscopy sample sizes and quality control.
- Comparison of scanned and non-scanned participants.

05_covariates_and_validity.py

Analyses:
- Probable PTSD across exposure bands, and its association with injury burden and delayed recall.
- Performance validity (Supplementary Table 11).
- Analyses omitting participants with potential testing-validity concerns (Supplementary Table 12).

3. Figures (Python 3.12)

06_figure1.py
Memory outcomes across injury burden (Fig. 1).

07_figure2.py
Cognitive reserve and memory by injury burden (Fig. 2).

08_figure3.py
Cognitive reserve, posterior cingulate tNAA/tCr and memory (Fig. 3).

09_figureS1.py
Model fit under alternative codings of injury exposure (Fig. S1).

10_figureS2.py
Predicted delayed recall across cognitive reserve and posterior cingulate tNAA/tCr (Fig. S2).

Fig. 4 (representative voxel placement and spectra) was rendered from the scanner images and Osprey output and is not produced from the tabular data. The figures use Arial when it is installed and fall back to Liberation Sans otherwise.

Running the scripts:

The scripts read from a data/ folder next to them, print each result to the console, and write the tables to tables/ and the figures to figures/. Run them in order:

    pip install -r requirements.txt
    for f in [01]*_*.py; do python "$f"; done

Data:

Direct identifiers have been removed from the analysis dataset. The de-identified data supporting this study are available from the corresponding author upon reasonable request. Place these files in data/:

- **IPV_CognitiveReserve_DATA.csv:** the analysis dataset, with spectroscopy quality-control exclusions applied.
- **IPV_CognitiveReserve_DATA_original_MRS_inclusion.csv:** the same dataset with spectroscopy values before quality control.
- **IPV_BMI_medication.csv:** body mass index and medication use.
- **IPV_MRS_PCG_components.csv:** posterior cingulate tNAA and tCr.
