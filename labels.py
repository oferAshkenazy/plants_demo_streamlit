import os
dirs = sorted(os.listdir(r'C:\Users\oferg\Downloads\plantvillage dataset\color'))
with open('models/labels.txt','w',encoding='utf-8') as f:
    f.write('|'.join(dirs))