import pandas as pd
from features_mapping import ClinicalDataExtractor
from Rules import InfectionChecker

print("Read .csv files...")
chartevent = pd.read_csv(r'D:\AI4LIFE\Y_te\mimic-iv-3.1\icu\chartevents.csv', nrows = 10000)
discharge = pd.read_csv(r'D:\AI4LIFE\Y_te\mimic-iv-note-deidentified-free-text-clinical-notes-2.2\note\discharge.csv', nrows = 10000)
microbiologyevent = pd.read_csv(r'D:\AI4LIFE\Y_te\mimic-iv-3.1\hosp\microbiologyevents.csv', nrows = 10000)
procedureevent = pd.read_csv(r'D:\AI4LIFE\Y_te\mimic-iv-3.1\icu\procedureevents.csv', nrows = 10000)
radiologyevent = pd.read_csv(r'D:\AI4LIFE\Y_te\mimic-iv-note-deidentified-free-text-clinical-notes-2.2\note\radiology_detail.csv', nrows = 10000)
icu_stay = pd.read_csv(r'D:\AI4LIFE\Y_te\mimic-iv-3.1\icu\icustays.csv', nrows=10000)
labevent = pd.read_csv('D:\AI4LIFE\Y_te\mimic-iv-3.1\hosp\labevents.csv', nrows = 10000)

mimic_data = {
        "chartevents": chartevent,
        "labevents": labevent,
        "discharge": discharge,
        "microbiology": microbiologyevent,
        "procedureevents": procedureevent,
        "radiology": radiologyevent
}
extractor = ClinicalDataExtractor(data_tables=mimic_data)
checker = InfectionChecker(extractor=extractor)

print("Processing...")
for row in icu_stay.itertuples():
    subject_id = row.subject_id
    intime = row.intime
    outtime = row.outtime

    vap_check = checker.check_vap
    