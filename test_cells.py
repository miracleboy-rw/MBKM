import matplotlib
matplotlib.use('Agg')
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

nb = json.load(open('MBKM_2_Ivan_Nur_Azmi.ipynb', encoding='utf-8'))
for idx in [18, 20, 22, 23]:
    cell = nb['cells'][idx]
    code = ''.join(cell.get('source', []))
    exec(code, globals())
print('Cells 18, 20, 22, 23 executed successfully!')
