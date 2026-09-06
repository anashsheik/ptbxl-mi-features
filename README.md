# Myocardial Infarction Classification with Classical Machine Learning

This project is a collaboration between OsloMet and SimulaMet. This project develops classical machine learning models to detect and classify myocardial infarction (MI) recordings versus normal (NORM) recordings from a PTB-XL dataset, using NeuroKit2 and 142 manually constructed ECG features.



**Author:** Anas Sheik  
**Supervisors:** Hugo L. Hammer (OsloMet) and Vajira Thambawita (SimulaMet)   
**Date:** May 2026  




To see the full report on this project, please see the bachelor thesis report submitted to OsloMet in May 2026.

## Project summary

The best-performing model was Random Forest (RF) on all 142 features.

- F1 = 0.7744
- AUROC = 0.9360
- Recall = 65.6 % (103 of 157 MI recordings)
- Precision = 94.5 % (103 of 109 positive predictions)

The most important single features from permutation importance came from QRS morphology
- `V3_R_amplitude`
- `V2_R_amplitude`
- `V2_QRS_duration`  

All three features are critical and reflect anteroseptal infarction.

## Full feature dataset

`features_full.csv` contains the extracted features for all 14,358 successfully
processed recordings, including the rows with missing (NaN) values that were
dropped from the complete-case analysis. Of these, 6,540 have a complete
142-feature vector. A further 218 recordings failed delineation entirely and
could not be processed, so they are not included; relative to the full
14,576 age-filtered cohort this gives the 44.9% coverage reported in the paper.
Columns: `ecg_id`, the 142 features, `strat_fold`, and `ground_truth`.


## Requirements

- Python 3.10 or newer
- `numpy`
- `pandas`
- `wfdb`
- `neurokit2`
- `scikit-learn`
- `matplotlib`


```bash
pip install -r requirements.txt
```


## How to use

### Step 1: PTB-XL dataset - `feature_extraction.py`

1. Download the PTB-XL dataset from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/).
2. Point the code at your local copy, either by editing `PTBXL_ROOT` in `feature_extraction.py` or by setting it as an environment variable:
   ```bash
   export PTBXL_ROOT=/path/to/ptb-xl-1.0.3
   ```
3. Run the file:
```bash
   python feature_extraction.py
```
   This produces: `train.csv`, `val.csv`, `test.csv`. 
   
**NOTE:** Runtime may differ and take some time depending on hardware.



### Step 2: Train & evaluate the model - `ml_pipeline.py`

```bash
python ml_pipeline.py
```
  This produces: `ml_results_individual_rounds.csv` and `permutation_importance_rf.csv`




## References
Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F. I., Samek, W., & Schaeffter, T. (2020). PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*, 7, 154. https://doi.org/10.1038/s41597-020-0495-6


