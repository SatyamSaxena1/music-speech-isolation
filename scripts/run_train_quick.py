import sys, os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from speech_isolation.train import train
m = train('configs/config.yaml', quick_run=True)
print('Returned model:', type(m))
