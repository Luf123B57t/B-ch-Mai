import pandas as pd
from features_mapping import ClinicalDataExtractor
from Rules import InfectionChecker, process_time_without_year

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
checker = InfectionChecker(extractor=extractor, verbose= False)

print("Processing...")
results = {
    "VAP": {"positive": 0, "negative": 0},
    "CLABSI": {"positive": 0, "negative": 0},
    "CAUTI": {"positive": 0, "negative": 0}
}

for row in icu_stay.itertuples():
    subject_id = row.subject_id
    intime = process_time_without_year(row.intime)
    outtime = process_time_without_year(row.outtime)
    stay_id = row.stay_id
    print("="*50)
    print("Processing patient {}....".format(subject_id))

    vap_check = checker.check_vap_subject(subject_id=subject_id, stay_id=stay_id, in_time = intime, out_time = outtime)
    if vap_check['final_vap']:
        results["VAP"]["positive"] += 1
    else:
        results["VAP"]["negative"] += 1

    clabsi_check = checker.check_clabsi_subject(subject_id=subject_id, stay_id=stay_id, in_time = intime, out_time = outtime)
    if clabsi_check['final_clabsi']:
        results["CLABSI"]["positive"] += 1
    else:
        results["CLABSI"]["negative"] += 1

    cauti_check = checker.check_cauti_subject(subject_id=subject_id, stay_id=stay_id, in_time = intime, out_time = outtime)
    if cauti_check['final_cauti']:
        results["CAUTI"]["positive"] += 1
    else:
        results["CAUTI"]["negative"] += 1
    print()


print("\n--- KẾT QUẢ THỐNG KÊ ---")
for infection_type, stats in results.items():
    print(f"{infection_type}: {stats['positive']} Ca Mắc | {stats['negative']} Ca Không")
    