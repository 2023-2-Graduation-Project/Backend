import pandas as pd
import time
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import csv
import warnings

# 경고메세지 끄기
warnings.filterwarnings(action='ignore')

################################### prepare #############################
hi = input("스팸 검사를 실시할 메세지를 입력하세요 : ")
hi=hi.replace(',','')

temp=[]
temp.append(hi)
df=pd.DataFrame(temp, columns=["text"])
df.to_csv("./test_data.csv", index=True, index_label="id")

print("검사중입니다 ...\n")
time.sleep(2)

################################### test ################################
class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        return text

text=[]
with open('./test_data.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    for i in reader:
        if i[1]=="text":
            continue
        text.append(i[1])

test_dataset = TextDataset(text)
test_data_loader = DataLoader(test_dataset, batch_size=1,shuffle=False)

model=torch.load("./model.pt", map_location=torch.device('cpu'))
model.eval()

with torch.no_grad():
    for x in test_data_loader:
        x=model.embed_texts(x[0])
        x=torch.tensor(x)
        result=model(x)
        result=float(result)

yes=round(result*100,2)
no=round(100-yes,2)

yes+=30
no-=30
yes=round(yes,2)
no=round(no,2)
print(f"[정상확률 : 스팸확률] ===> [{no}% : {yes}%]")

############################# new_word_method ################################
normal_neo=pd.read_csv('normal_neo.csv')
normal_neo=list(normal_neo)
normal_detect=''
normal_cnt=1
spam_neo=pd.read_csv('spam_neo.csv')
spam_neo=list(spam_neo)
spam_detect=''
spam_cnt=1

input_word = hi.split(' ')
for word in input_word:
    if word in normal_neo:
        normal_cnt+=1
        normal_detect+=word
        normal_detect+=', '
    if word in spam_neo:
        spam_cnt+=1
        spam_detect+=word
        spam_detect+=', '

print("\n적발당한 정상 신조어 : ",normal_detect[:-2])
print("적발당한 스팸 신조어 : ",spam_detect[:-2])

yes=yes*spam_cnt
no=no*normal_cnt

Y=round(yes*100/(yes+no),2)
N=round(no*100/(yes+no),2)
print(f"보정된 [정상확률 : 스팸 확률] ===> [{N}% : {Y}%]\n")

################################### print result ##################################
if Y>50:
    print("삐빅! 스팸입니다.")
else:
    print("정상입니다.")