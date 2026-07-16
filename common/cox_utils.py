"""
Shared Cox endpoint label and category mappings.
Imported by plot_combined_figure.py and plot_cox_model_comparison.py.
"""

# Maps raw long ICD-10/HES event column names → short display labels.
COX_EVENT_SHORT_NAMES = {
    'Other Chronic Obstructive Pulmonary Disease':         'COPD',
    'Non-Insulin-Dependent Diabetes Mellitus':             'Type 2 Diabetes',
    'Acute Myocardial Infarction':                         'Myocardial Infarction',
    'Chronic Ischaemic Heart Disease':                     'Ischaemic Heart Disease',
    'Atrial Fibrillation And Flutter':                     'Atrial Fibrillation',
    'Other Intervertebral Disk Disorders':                 'Intervertebral Disk Disease',
    'Osteoporosis Without Pathological Fracture':          'Osteoporosis',
    'Other Diseases Of Liver':                             'Liver Disease',
    'Other Rheumatoid Arthritis':                          'Rheumatoid Arthritis',
    'Other Hypothyroidism':                                'Hypothyroidism',
    'Other Arthrosis':                                     'Arthrosis',
    'Coxarthrosis':                                        'Hip Arthrosis',
    'Coxarthrosis [Arthrosis Of Hip]':                     'Hip Arthrosis',
    'Gonarthrosis':                                        'Knee Arthrosis',
    'Gonarthrosis [Arthrosis Of Knee]':                    'Knee Arthrosis',
    'Thyrotoxicosis [Hyperthyroidism]':                    'Hyperthyroidism',
    'Other Spondylopathies':                               'Spondylopathy',
    'Stroke, Not Specified As Haemorrhage Or Infarction':  'Stroke',
    "Alzheimer'S Disease":                                 "Alzheimer's",
    "Parkinson'S Disease":                                 "Parkinson's",
}

# Maps display labels (post-shortening) → organ-system category int.
COX_EVENT_CATEGORIES = {
    'Osteoporosis': 1, 'Knee Arthrosis': 1, 'Hip Arthrosis': 1,
    'Spondylopathy': 1, 'Arthrosis': 1, 'Rheumatoid Arthritis': 1,
    'Intervertebral Disk Disease': 1,
    'Obesity': 2, 'Type 2 Diabetes': 2, 'Hypothyroidism': 2, 'Hyperthyroidism': 2,
    'Heart Failure': 3, 'Atrial Fibrillation': 3, 'Ischaemic Heart Disease': 3,
    'Angina Pectoris': 3, 'Myocardial Infarction': 3,
    'COPD': 4, 'Asthma': 4,
    'Chronic Renal Failure': 5, 'Acute Renal Failure': 5, 'Liver Disease': 5,
    'Dementia': 6, "Alzheimer's": 6, "Alzheimer's Disease": 6,
    "Parkinson's": 6, 'Parkinsonism': 6, 'Cerebral Infarction': 6, 'Stroke': 6,
    'Death': 7, 'All-Cause Death': 7, 'Cancer Death': 7,
}

# Full category names (for legends with space to spare)
COX_CATEGORY_NAMES = {
    1: 'Musculoskeletal', 2: 'Metabolic', 3: 'Cardiovascular',
    4: 'Respiratory', 5: 'Hepatic & Renal', 6: 'Neurological', 7: 'Mortality',
}

# Abbreviated category names (for tight axis labels)
COX_CATEGORY_NAMES_ABBREV = {
    1: 'Musculosk.', 2: 'Metabolic', 3: 'Cardiovasc.',
    4: 'Respiratory', 5: 'Hep. & Renal', 6: 'Neurolog.', 7: 'Mortality',
}
