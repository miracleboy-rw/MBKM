import re
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ClinicalFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Transformer medis bebas bocor untuk mengekstrak 7 sindrom klinis murni
    serta kategori mikrobiologi & spesimen dari teks rekam medis mentah.
    Variabel bias sumber data (hipertensi, hospital origin, riwayat antibiotik) telah dieliminasi.
    """
    def __init__(self):
        self.top_spesimen = ['urin', 'sputum', 'darah', 'jaringan', 'abses']
        
    def fit(self, X, y=None):
        return self
        
    def _clean_patogen(self, val):
        v = str(val).lower() if pd.notna(val) else 'lainnya'
        if 'klebsiella' in v: return 'klebsiella'
        elif 'coli' in v or 'escherichia' in v: return 'e_coli'
        elif 'acinetobacter' in v: return 'acinetobacter'
        elif 'staph' in v and ('aureus' in v or 'coag +' in v): return 'staph_aureus'
        elif 'staph' in v: return 'staph_cons'
        elif 'yeast' in v or 'candida' in v: return 'candida_yeast'
        elif 'pseudomonas' in v: return 'pseudomonas'
        else: return 'lainnya'
        
    def transform(self, X):
        df_out = pd.DataFrame(index=X.index)
        
        diag_clean = X['Diagnosis'].fillna('').astype(str).str.replace(
            r'with death due to brain injury[,\w\s]*', '', regex=True, case=False
        )
        kom_clean = X['Komorbid'].fillna('').astype(str)
        text_comb = (diag_clean + ' ' + kom_clean).str.lower()
        
        df_out['is_respiratory_failure'] = text_comb.apply(lambda x: 1 if re.search(r'\b(respiratory failure|gagal napas|gagal nafas|ards|edema paru|pulmonary edema|pneumonia)\b', x) else 0)
        df_out['is_kidney_failure'] = text_comb.apply(lambda x: 1 if re.search(r'\b(ckd|aki|renal|azotemia|ginjal|kidney|hepatorenal|nephro\w*|nefr\w*)\b', x) else 0)
        df_out['is_sepsis'] = text_comb.apply(lambda x: 1 if re.search(r'\b(sepsis|septic|septik|septicemia|septikemia|syok\s+septik|septic\s+shock)\b', x) else 0)
        df_out['is_diabetes'] = text_comb.apply(lambda x: 1 if re.search(r'\b(niddm|iddm|diabetes|dm|hiperglikemia|hyperglycemia)\b', x) else 0)
        df_out['is_stroke_ich'] = text_comb.apply(lambda x: 1 if re.search(r'\b(ich|stroke|cerebri|cerebral|snh|intracranial|cva|subarachnoid)\b', x) else 0)
        df_out['is_cardiovascular'] = text_comb.apply(lambda x: 1 if re.search(r'\b(heart failure|gagal jantung|cardiogenic|infark miokard|myocardial infarction|cor pulmonale)\b', x) else 0)
        df_out['is_liver_disease'] = text_comb.apply(lambda x: 1 if re.search(r'\b(hepatic|liver|sirosis|cirrhosis|hepatorenal|hepatitis|hati)\b', x) else 0)
        
        df_out['spesimen_kategori'] = X['Jenis sampel'].apply(
            lambda val: str(val).strip().lower() if pd.notna(val) and str(val).strip().lower() in self.top_spesimen else 'lainnya'
        )
        df_out['patogen_kategori'] = X['Nama bakteri'].apply(self._clean_patogen)
        
        fitur_cols = [
            'is_respiratory_failure', 'is_kidney_failure', 'is_sepsis',
            'is_diabetes', 'is_stroke_ich', 'is_cardiovascular', 'is_liver_disease',
            'spesimen_kategori', 'patogen_kategori'
        ]
        return df_out[fitur_cols]
