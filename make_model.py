import pandas as pd
import numpy as np
from statistics import mean,stdev
import configparser
import os
import math
#「学習」
import torch
import torch.optim as optimizers
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import copy
config = configparser.ConfigParser()
config.read('./data/account.txt')
ex_pair = config['oanda']['pair']           
asi = config['oanda']['asi']
epochs = int(config['oanda']['epoch'])
bar = config['oanda']['bar']
filename = './data/'+ex_pair+'_'+asi+'.csv'
try:
    os.remove('./data/train_log'+'_'+ex_pair+'_'+asi+'_'+bar+'.csv')
except:
    pass
#MACDの値を返す関数を定義する。（引数はロウソク足の終値）
def macd_data(data):
    #短期と長期の指数平滑移動平均、MACDのリスト、signalのリスト
    Lema, Sema, MACD, signal = [], [], [], []
    num = 0
    num2 = 0
    d = []
    for m in data:
        d.append(m)

    for i in range(26, len(d)):
        if i == 26 and num == 0:
            Lema.append(mean(d[i-26:i]))
            Sema.append(mean(d[i-12:i]))
            MACD.append(Sema[num]-Lema[num])
            num+=1
        elif i != 26:
            Lema.append(Lema[num-1]*(1-(2/27))+d[i-1]*2/27)
            Sema.append(Sema[num-1]*(1-(2/13))+d[i-1]*2/13)
            MACD.append(Sema[num]-Lema[num])
            num+=1
    for i in range(9, len(MACD)):
        if i == 9 and num2 == 0:
            signal.append(mean(MACD[i-9:i]))
            num2+=1
        elif i != 9:
            signal.append(signal[num2-1]*(1-(2/10))+MACD[i-1]*2/10)
            num2+=1
    #MACDとsignalの値を返す
    return MACD, signal
#MACDのシグナルを計算
def macd_signal(MACD, signal):
    MACD_signal = []
    MACD = MACD[9:]
    for i in range(len(MACD)):
        MACD_signal.append(MACD[i] - signal[i])
    return MACD_signal
def make_df(df, df_all, bar):
    bar = int(bar)
    #label作成
    df_al = df_all[['close']]
    df_al2 = df_al.shift(-bar)
    sh = df_al2.shape
    df_al2.columns = range(sh[1]) 
    df_al3 = (df_al.iloc[:, 0] - df_al2.iloc[:, 0])
    df_al3 = df_al3.shift(-35)
    df_al3 = df_al3.dropna(how='any')
    df_label = df_al3
    #上、下、そのままに三分割
    idx_up = df_al3.index[df_al3 > 0]
    idx_down = df_al3.index[df_al3 < 0]
    idx_no = df_al3.index[df_al3 == 0]
    df_al3 = pd.DataFrame(df_al3, columns = ['diff'])
    df_al3 = df_al3.assign(label=0)
    df_al3.iloc[idx_up, 1] = 1
    df_al3.iloc[idx_down, 1] = 2
    df_up = df_al3[df_al3.iloc[:, 1] == 1]
    df_down = df_al3[df_al3.iloc[:, 1] == 2]
    #さらに分割して５つに分ける
    df_up = df_up.sort_values('diff', ascending=False)
    df_down = df_down.sort_values('diff', ascending=False)
    len_up = math.floor(len(df_up)/2)
    len_down = math.floor(len(df_down)/2)
    lim_up = str(df_up.iloc[len_up, 0])
    lim_down = str(df_down.iloc[len_down, 0])
    df_1 = df_up[len_up:]
    df_2 = df_up[:len_up-1]
    df_3 = df_down[len_down+1:]
    df_4 = df_down[:len_down]
    idx_1 = list(df_1.index)
    idx_2 = list(df_2.index)
    idx_3 = list(df_3.index)
    idx_4 = list(df_4.index)
    df_label[idx_1] = 1
    df_label[idx_2] = 2
    df_label[idx_3] = 3
    df_label[idx_4] = 4
    df_label[idx_no] = 5

    #MACDデータ作成
    for i in range(bar-1):
        if i == 0:
            df_shift = df
        df_shift = df_shift.shift(-1)
        df = pd.concat([df, df_shift], axis=1)
        
    #結合
    sh = df.shape
    df.columns = range(sh[1])
    df = df.dropna(how='any') 
    df = pd.concat([df_label, df], axis=1)
    sh = df.shape
    df.columns = range(sh[1])
    return df, lim_up, lim_down
#データの準備
df_all = pd.read_csv(filename)
df_clo = df_all['close']
MACD, signal = macd_data(df_clo)
#print(MACD, signal)
MACD_signal = macd_signal(MACD, signal)
df_mac = pd.Series(MACD_signal)
#print(df_mac)
df, lim_up, lim_down = make_df(df_mac, df_all, bar)
#モデルの利益確定範囲を指定
config['oanda']['limit_up'] = lim_up
config['oanda']['limit_down'] = lim_down
with open('./data/account.txt', 'w') as file:
    config.write(file)
#print(df)

#ラベル別分類する．
df_buy = df.loc[df.iloc[:, 0] == 1]
df_buy2 = df.loc[df.iloc[:, 0] == 2]
df_sell = df.loc[df.iloc[:, 0] == 3]
df_sell2 = df.loc[df.iloc[:, 0] == 4]
df_no = df.loc[df.iloc[:, 0] == 5]




#データの分割
def extract_train_vali_test(df, train_ratio=0.8, vali_ratio=0.1):
    num = df.shape[0]
    train_end = int(num * train_ratio)
    vali_end = int(num * (train_ratio + vali_ratio))
    return df.iloc[:train_end, :], df.iloc[train_end:vali_end, :], df.iloc[vali_end:, :]

df_buy_train, df_buy_vali, df_buy_test = extract_train_vali_test(df_buy)
df_buy2_train, df_buy2_vali, df_buy2_test = extract_train_vali_test(df_buy2)
df_sell_train, df_sell_vali, df_sell_test = extract_train_vali_test(df_sell)
df_sell2_train, df_sell2_vali, df_sell2_test = extract_train_vali_test(df_sell2)
df_no_train, df_no_vali, df_no_test = extract_train_vali_test(df_no)

#各分割データを結合
def combine_buy_sell_no(df_buy, df_buy2, df_sell, df_sell2, df_no):
    num_buy = df_buy.shape[0]
    num_buy2 = df_buy2.shape[0]
    num_sell = df_sell.shape[0]
    num_sell2 = df_sell2.shape[0]
    num_no = df_no.shape[0]
    label = ([0] * num_buy) + ([1] * num_buy2) + ([2] * num_sell) + ([3] * num_sell2) + ([4] * num_no)
    df_all = pd.concat([df_buy, df_buy2, df_sell, df_sell2, df_no], axis=0)
    df_label = pd.DataFrame(dict(label=label), index=df_all.index)
    return pd.concat([df_label, df_all], axis=1)

df_train = combine_buy_sell_no(df_buy_train, df_buy2_train, df_sell_train, df_sell2_train, df_no_train)
df_vali = combine_buy_sell_no(df_buy_vali, df_buy2_vali, df_sell_vali, df_sell2_vali, df_no_vali)
df_test = combine_buy_sell_no(df_buy_test, df_buy2_test, df_sell_test, df_sell2_test, df_no_test)
"""
df_train = df_train.iloc[:, 1:]
df_vali = df_vali.iloc[:, 1:]
df_test = df_test.iloc[:, 1:]
"""


#dfから特徴量とラベルのみ抽出
def extract_x_y(df: pd.DataFrame):
    x = df.iloc[:, 2:] #特徴量
    y = df.iloc[:, 0] #ラベル
    return x, y

x_train, y_train = extract_x_y(df_train)
x_vali, y_vali = extract_x_y(df_vali)
x_test, y_test = extract_x_y(df_test)
"""
print(x_train)
print(x_vali)
print(x_test)
"""
def make_data(x, y): #ラベルと特徴量を結合してリストにする
    x = x.astype(np.float32) #astype() : ndarray要素のデータ型を変更
    y = y.astype(np.int64)
    data = []
    for xx, yy in zip(x, y):
        data.append((xx, yy))
    return data

data_train = make_data(x_train.to_numpy(), y_train.to_numpy())
data_vali = make_data(x_vali.to_numpy(), y_vali.to_numpy())
data_test = make_data(x_test.to_numpy(), y_test.to_numpy())


#データローダの設定
train_dataloader = DataLoader(data_train, batch_size=5, shuffle=True)
vali_dataloader = DataLoader(data_vali, batch_size=5, shuffle=False)


#学習
class Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=500):
        #print(input_dim, output_dim)
        super(Model, self).__init__()
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.d1 = nn.Dropout(p=0.2)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.d2 = nn.Dropout(p=0.2)
        self.a2 = nn.ReLU()
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.d3 = nn.Dropout(p=0.2)
        self.a3 = nn.ReLU()
        self.l4 = nn.Linear(hidden_dim, hidden_dim)
        self.d4 = nn.Dropout(p=0.2)
        self.a4 = nn.ReLU()
        self.l5 = nn.Linear(hidden_dim, hidden_dim)
        self.d5 = nn.Dropout(p=0.2)
        self.a5 = nn.ReLU()
        self.l6 = nn.Linear(hidden_dim, output_dim)
        """
        self.layers = [self.l1, self.d1, self.a1,
                        self.l2, self.d2, self.a2,
                        self.l3, self.d3, self.a3,
                        self.l4]
        self.layers = [self.l1, self.a1,
                        self.l2, self.a2,
                        self.l4]
        """
        self.layers = [self.l1, self.a1, self.l2, self.a2, self.l3, self.a3, self.l4, self.a4, self.l5, self.a5, self.l6]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x_dim = x_train.shape[1] #入力要素数(特徴量の要素数)
y_dim = 5 #出力要素数
model = Model(x_dim, y_dim).to(device) #モデルの定義

criterion = nn.CrossEntropyLoss() #損失関数
optimizer = optimizers.Adam(model.parameters()) #最適化器（オプティマイザ）：誤差が最小になるようにパラメータを調整するプロセス　最適化関数は複数種類存在（SGD, ADAM）

def compute_loss(t, y): #損失関数の計算
    return criterion(y, t)

def train_step(x, t):
    model.train() #ネットワークを学習モードに
    preds = model(x) #訓練データxから出力を求める
    loss = compute_loss(t, preds) #出力と正解の誤差を求める
    optimizer.zero_grad() #勾配のリセット
    loss.backward() #誤差逆伝播
    optimizer.step() #パラメータの最適化

    return loss, preds

def vali_step(x, t):
    model.eval() #ネットワークを推論モードに
    preds = model(x)
    loss = criterion(preds, t)

    return loss, preds

log = dict(epoch=[], train_loss=[], train_acc=[], vali_loss=[], vali_acc=[])

best_loss = 1e+10
best_model_params = None

for epoch in range(epochs):

#学習
    train_loss = 0.
    train_acc = 0.

    for (x, t) in train_dataloader:
        x, t = x.to(device), t.to(device)
        loss, preds = train_step(x, t)
        train_loss += loss.item()
        train_acc += \
                accuracy_score(t.tolist(),
                preds.argmax(dim=-1).tolist())

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader)

#検証
    vali_loss = 0.
    vali_acc = 0.

    for (x, t) in vali_dataloader:
        x, t = x.to(device), t.to(device)
        loss, preds = vali_step(x, t)
        #print(loss)
        #print(preds)
        vali_loss += loss.item()
        vali_acc += \
            accuracy_score(t.tolist(),preds.argmax(dim=-1).tolist())

    vali_loss /= len(vali_dataloader)
    vali_acc /= len(vali_dataloader)
#epochごとの結果をlogに記録
    if vali_loss < best_loss:
        print("best loss updated")
        # preserve the best parameters
        best_model_params = copy.deepcopy(model.state_dict())
        best_loss = vali_loss
        torch.save(best_model_params, 'FXmodel'+'_'+ex_pair+'_'+asi+'_'+bar+'.pth')
    log['epoch'].append(epoch+1)
    log['train_loss'].append(train_loss)
    log['train_acc'].append(train_acc)
    log['vali_loss'].append(vali_loss)
    log['vali_acc'].append(vali_acc)
    pd.DataFrame(log).to_csv('./data/train_log'+'_'+ex_pair+'_'+asi+'_'+bar+'.csv')

