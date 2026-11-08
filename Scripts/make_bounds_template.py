import os
import argparse
import pandas as pd

def get_mimic_units(mimic_path):
    units = {}
    
    # Read lab units from labevents (just 1M rows should cover it)
    lab_file = os.path.join(mimic_path, 'hosp', 'labevents.csv.gz')
    try:
        df = next(pd.read_csv(lab_file, chunksize=1000000, compression='gzip'))
        uoms = df.groupby('itemid')['valueuom'].first()
        units.update(uoms.to_dict())
    except:
        pass
        
    # Read vital units from chartevents
    chart_file = os.path.join(mimic_path, 'icu', 'chartevents.csv.gz')
    try:
        df = next(pd.read_csv(chart_file, chunksize=1000000, compression='gzip'))
        uoms = df.groupby('itemid')['valueuom'].first()
        units.update(uoms.to_dict())
    except:
        pass
        
    return units

def get_eicu_units(eicu_path):
    units = {}
    
    lab_file = os.path.join(eicu_path, 'lab.csv.gz')
    try:
        df = next(pd.read_csv(lab_file, chunksize=1000000))
        uoms = df.groupby('labname')['labmeasurenamesystem'].first()
        units.update(uoms.to_dict())
    except:
        pass
        
    return units

def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    OUT_FILE = os.path.join(PROJECT_ROOT, "Results", "final", "bounds_template.csv")
        
    LAB_ITEMS = {
        'Albumin':          {'mimic': [50862], 'eicu': 'albumin'},
        'AnionGap':         {'mimic': [50868], 'eicu': 'anion gap'},
        'Bicarbonate':      {'mimic': [50882], 'eicu': 'bicarbonate'},
        'BUN':              {'mimic': [51006], 'eicu': 'BUN'},
        'Calcium':          {'mimic': [50893, 50804], 'eicu': 'calcium'},
        'Chloride':         {'mimic': [50902, 50806], 'eicu': 'chloride'},
        'Creatinine':       {'mimic': [50912], 'eicu': 'creatinine'},
        'Glucose':          {'mimic': [50931, 50809], 'eicu': 'glucose'},
        'Sodium':           {'mimic': [50983, 50824], 'eicu': 'sodium'},
        'Potassium':        {'mimic': [50971, 50822], 'eicu': 'potassium'},
        'Lactate':          {'mimic': [50813], 'eicu': 'lactate'},
        'Hemoglobin':       {'mimic': [51222], 'eicu': 'Hgb'},
        'Hematocrit':       {'mimic': [51221], 'eicu': 'Hct'},
        'Platelets':        {'mimic': [51265], 'eicu': 'platelets x 1000'},
        'WBC':              {'mimic': [51301, 51300], 'eicu': 'WBC x 1000'},
        'Lymphocytes':      {'mimic': [51244, 51245], 'eicu': '-lymphs'},
        'Neutrophils':      {'mimic': [51256], 'eicu': '-polys'},
        'INR':              {'mimic': [51237], 'eicu': 'PT - INR'},
        'PT':               {'mimic': [51274], 'eicu': 'PT'},
        'PTT':              {'mimic': [51275], 'eicu': 'PTT'},
        'ALT':              {'mimic': [50861], 'eicu': 'ALT (SGPT)'},
        'AST':              {'mimic': [50878], 'eicu': 'AST (SGOT)'},
        'Bilirubin_Total':  {'mimic': [50885], 'eicu': 'total bilirubin'},
        'pH':               {'mimic': [50820], 'eicu': 'pH'},
        'pO2':              {'mimic': [50821], 'eicu': 'paO2'},
        'pCO2':             {'mimic': [50818], 'eicu': 'paCO2'},
        'BaseExcess':       {'mimic': [50802], 'eicu': 'Base Excess'},
        'Magnesium':        {'mimic': [50960], 'eicu': 'magnesium'},
        'Phosphate':        {'mimic': [50970], 'eicu': 'phosphate'},
        'LDH':              {'mimic': [50954], 'eicu': 'LDH'},
        'Troponin_T':       {'mimic': [51003], 'eicu': 'troponin - T'},
        'CRP':              {'mimic': [50889], 'eicu': 'CRP'},
    }
    
    VITAL_ITEMS = {
        'HR':     {'mimic': [220045], 'eicu': 'heartrate'},
        'RR':     {'mimic': [220210, 224690], 'eicu': 'respiration'},
        'SBP':    {'mimic': [220179, 220050], 'eicu': 'systemicsystolic'},
        'DBP':    {'mimic': [220180, 220051], 'eicu': 'systemicdiastolic'},
        'MAP':    {'mimic': [220052, 220181, 225312], 'eicu': 'systemicmean'},
        'Temp':   {'mimic': [223762, 223761], 'eicu': 'temperature'},
        'O2Sat':  {'mimic': [220277], 'eicu': 'sao2'},
        'GCS':    {'mimic': [220739, 223900, 223901], 'eicu': 'GCS'},
    }
    
    parser = argparse.ArgumentParser(description="Create bounds template.")
    parser.add_argument("--mimic-dir", type=str, required=True, help="Path to raw MIMIC-IV folder")
    parser.add_argument("--eicu-dir", type=str, required=True, help="Path to raw eICU folder")
    args = parser.parse_args()

    mimic_units = get_mimic_units(args.mimic_dir)
    eicu_units = get_eicu_units(args.eicu_dir)
    
    rows = []
    
    for items in [VITAL_ITEMS, LAB_ITEMS]:
        for var, mapping in items.items():
            m_uoms = [str(mimic_units.get(i, '')) for i in mapping['mimic']]
            m_uom = ' / '.join(list(set(filter(None, m_uoms))))
            
            e_uom = str(eicu_units.get(mapping['eicu'], ''))
            
            rows.append({
                'variable': var,
                'unit_mimic': m_uom,
                'unit_eicu': e_uom,
                'lower': '',
                'upper': '',
                'unit': '',
                'source': ''
            })
            
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    df.to_csv(OUT_FILE, index=False)
    print(f"Created {OUT_FILE} with {len(df)} rows.")

if __name__ == "__main__":
    main()
