import os

def path(str):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),str)

data_load_path={
    '密度':       path('data/density.txt'),
    '电导率':      path('data/ec.txt'),
    '热导率':      path('data/tc.txt'),
    '粘度':       path('data/viscosity.txt'),
    '硬度':       path('data/hardness.txt')
}
model_save_path={
    '密度':       path('save/density'),
    '电导率':      path('save/ec'),
    '热导率':      path('save/tc'),
    '粘度':       path('save/viscosity'),
    '硬度':       path('save/hardness')
}