import sys, os, importlib, traceback
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
try:
    infer_mod = importlib.import_module('speech_isolation.infer')
    print('imported infer')
    m = importlib.import_module('speech_isolation.models')
    print('imported models')
    Model = m.IdentitySeparator
    model = Model()
    out = infer_mod.infer(model, 'test_assets/mixture.wav', 'out_test')
    print('infer returned', out)
except Exception:
    traceback.print_exc()
    raise
