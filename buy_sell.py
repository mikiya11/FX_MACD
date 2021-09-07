from statistics import mean,stdev
import configparser
import pandas as pd
import pytz
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.trades as trades
from oandapyV20.exceptions import V20Error
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import copy
import time



config = configparser.ConfigParser()
config.read('./data/account.txt')
account_id = config['oanda']['account_id']  # ID
api_key = config['oanda']['api_key']        #トークンパス
ex_pair = config['oanda']['pair']           #対象通貨
lot = config['oanda']['lot']                #lot数 
asi = config['oanda']['asi']                #取得した時間足

#modelパス
loa_path = 'FXmodel'+'_'+ex_pair+'_'+asi+'.pth'
api = oandapyV20.API(access_token=api_key, environment="live")

def get_mdata(ex_pair, api, asi):
    params1 = {"instruments": ex_pair}
    psnow = pricing.PricingInfo(accountID=account_id, params=params1)
    try:
        now = api.request(psnow) #現在の価格を取得
    except V20Error:
        now = api.request(psnow) #現在の価格を取得
    end = now['time']
    params = {"count":100,"granularity":asi,"to":end}
    r = instruments.InstrumentsCandles(instrument=ex_pair, params=params,)
    try:
        apires = api.request(r)
    except V20Error:
        apires = api.request(r)
    res = r.response['candles']
    end = res[0]['time']
    n = 0
    res1 = res

    data = []
    price = []
    #形を成形
    for raw in res1:
        data.append([raw['time'], raw['mid']['o'], raw['mid']['h'], raw['mid']['l'], raw['mid']['c']])
    #DataFrameに変換
    df = pd.DataFrame(data)
    df.columns = ['date', 'open', 'high', 'low', 'close']
    
    #時間を全て日本時間に変更する。
    for i in df['date']:
        i = pd.Timestamp(i).tz_convert('Asia/Tokyo')

    df.iloc[:, 0] = df.iloc[:, 0].astype('datetime64[ns]')
    df.iloc[:, 1:] = df.iloc[:, 1:].astype('float')   
    return df

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
def make_df(df, df_rev):  
    df_rev = df_rev.drop('date', axis=1)
    #label作成
    df_20 = df_rev[19::20]
    df_al = df_20.reset_index(drop=True)
    df_al.insert(0, 'label', 0)
    for i in range(len(df_al)-1):
        if df_al.iloc[i, 1] < df_al.iloc[i+1, 1] and abs(df_al.iloc[i, 1] - df_al.iloc[i+1, 1]) > 0.04:
            df_al.iloc[i, 0] = 1
        elif df_al.iloc[i, 1] < df_al.iloc[i+1, 1] and abs(df_al.iloc[i, 1] - df_al.iloc[i+1, 1]) <= 0.04:
            df_al.iloc[i, 0] = 2
        elif df_al.iloc[i, 1] > df_al.iloc[i+1, 1] and abs(df_al.iloc[i, 1] - df_al.iloc[i+1, 1]) > 0.04:
            df_al.iloc[i, 0] = 3
        elif df_al.iloc[i, 1] > df_al.iloc[i+1, 1] and abs(df_al.iloc[i, 1] - df_al.iloc[i+1, 1]) <= 0.04:
            df_al.iloc[i, 0] = 4
        elif df_al.iloc[i, 1] == df_al.iloc[i+1, 1]:
            df_al.iloc[i, 0] = 5
    df_al = df_al.drop(df_al.index[[0]])
    df_al = df_al.drop(df_al.index[[0]])
    df_al = df_al.reset_index(drop=True)
    #print(df_al)
    #MACDデータ作成
    df_1 = df[::20]
    df_2 = df[1::20]
    df_3 = df[2::20]
    df_4 = df[3::20]
    df_5 = df[4::20]
    df_6 = df[5::20]
    df_7 = df[6::20]
    df_8 = df[7::20]
    df_9 = df[8::20]
    df_10 = df[9::20]
    df_11 = df[10::20]
    df_12 = df[11::20]
    df_13 = df[12::20]
    df_14 = df[13::20]
    df_15 = df[14::20]
    df_16 = df[15::20]
    df_17 = df[16::20]
    df_18 = df[17::20]
    df_19 = df[18::20]
    df_20 = df[19::20]

    df_1 = df_1.reset_index(drop=True)
    df_2 = df_2.reset_index(drop=True)
    df_3 = df_3.reset_index(drop=True)
    df_4 = df_4.reset_index(drop=True)
    df_5 = df_5.reset_index(drop=True)
    df_6 = df_6.reset_index(drop=True)
    df_7= df_7.reset_index(drop=True)
    df_8 = df_8.reset_index(drop=True)
    df_9 = df_9.reset_index(drop=True)
    df_10 = df_10.reset_index(drop=True)
    df_11 = df_11.reset_index(drop=True)
    df_12 = df_12.reset_index(drop=True)
    df_13 = df_13.reset_index(drop=True)
    df_14 = df_14.reset_index(drop=True)
    df_15 = df_15.reset_index(drop=True)
    df_16 = df_16.reset_index(drop=True)
    df_17= df_17.reset_index(drop=True)
    df_18 = df_18.reset_index(drop=True)
    df_19 = df_19.reset_index(drop=True)
    df_20 = df_20.reset_index(drop=True)
    df_a = df_al.iloc[:, 0] 
    df = pd.concat([df_a, df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8, df_9, df_10, df_11, df_12, df_13, df_14, df_15, df_16, df_17, df_18, df_19, df_20], axis='columns')
    sh = df.shape
    df.columns = range(sh[1])
    return df
#dfから特徴量とラベルのみ抽出
def extract_x_y(df: pd.DataFrame):
    x = df.iloc[:, 1:] #特徴量
    y = df.iloc[:, 0] #ラベル
    return x, y

#買い
def buy_signal(now_price, flag, account_id, api, b, s, a, lot, times):
    lot = str(lot)
    b.append(now_price)
    data = {
         "order": {
            "instrument": "USD_JPY",
            "units": "+"+lot,
            "type": "MARKET",
        }
    }
    ticket = orders.OrderCreate(account_id, data=data)
    try:
        api.request(ticket)
    except V20Error:
        api.request(ticket)
    times = 0
    flag["buy_signal"] = 1
    print("b")
    return b, times, flag
#売り
def sell_signal(now_price, flag, account_id, api, b, s, a, lot, times):
    lot = str(lot)
    s.append(now_price)
    data = {
         "order": {
             "instrument": "USD_JPY",
             "units": "-"+lot,
            "type": "MARKET",
        }
    }
    ticket = orders.OrderCreate(account_id, data=data)
    try:
        api.request(ticket)
    except V20Error:
        api.request(ticket)
    times = 0
    flag["sell_signal"] = 1
    print("s")
    return s, times, flag
#決済
def close_signal(now_price, flag, account_id, api, b, s, a, lot, times):
    i = 0
    j = 0
    lot = float(lot)
    if flag["buy_signal"] == 1:
        data = {"longUnits":"ALL"}
        ticket = positions.PositionClose(accountID=account_id, instrument="USD_JPY", data=data)
        try:
            api.request(ticket)
        except V20Error:
            api.request(ticket)
        times = 0
        flag["buy_signal"] = 0
        while i < len(b):
            d = b[i]
            #print(now_price, "-", d)
            a += now_price*lot - d*lot - 0.008*lot
            i += 1      
        
    if flag["sell_signal"] == 1:
        data = {"shortUnits":"ALL"}
        ticket = positions.PositionClose(accountID=account_id, instrument="USD_JPY", data=data)
        try:
            api.request(ticket)
        except V20Error:
            api.request(ticket)
        times = 0
        while j < len(s):
            d = s[j]
            #print(d, "-", now_price)
            a += d*lot - now_price*lot - 0.008*lot
            j += 1
        flag["sell_signal"] = 0
    print(a)
    if a < -50000:
        print("fail")
        while a < -50000:
            a = a
    y = []
    return y, a, times, flag

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

def make_data(x, y): #ラベルと特徴量を結合してリストにする
    x = x.astype(np.float32) #astype() : ndarray要素のデータ型を変更
    y = y.astype(np.int64)
    data = []
    for xx, yy in zip(x, y):
        data.append((xx, yy))
    return data

#メイン
flag = {"buy_signal" : 0,
        "sell_signal" : 0,
        "gold_signal" : 0,
        "dead_signal" : 0,
        "good_signal" : 0,
        "bad_signal" :0
	}
b = []
s = []
a = 0
model = Model(20, 5).to(device)
model.load_state_dict(torch.load(loa_path))
model.eval()
times = 240
old_price = 0
while True:
    df_all = get_mdata(ex_pair, api, asi)
    #print(df_all)
    now_price = df_all.iloc[99, 4]
    #print(now_price)
    if old_price == 0:
        old_price = now_price
    if abs(now_price - old_price) > 10:
        if flag["sell_signal"] == 1:
            s, a = close_signal(now_price, flag, account_id, api, b, s, a, lot)
            flag["sell_signal"] = 0
        if flag["buy_signal"] == 1:
             b, a = close_signal(now_price, flag, account_id, api, b, s, a, lot)
             flag["buy_signal"] = 1
    if now_price - old_price > 0.04 and flag["buy_signal"] == 1:
        b, a, times, flag = close_signal(now_price, flag, account_id, api, b, s, a, lot, times)
        times = 240
    elif old_price - now_price > 0.04 and flag["sell_signal"] == 1:
        s, a, times, flag = close_signal(now_price, flag, account_id, api, b, s, a, lot, times)
        times = 240
    if times == 240:
        df_all = get_mdata(ex_pair, api, asi)
        #print(df_all)
        old_price = df_all.iloc[99, 4]
        df_clo = df_all['close']
        MACD, signal = macd_data(df_clo)
        MACD_signal = macd_signal(MACD, signal)
        df_mac = pd.Series(MACD_signal)
        df_mac = df_mac.iloc[5:]
        df = make_df(df_mac, df_all)
        sh = df.shape
        df.columns = range(sh[1])
        df = df.drop(0)
        df = df.drop(1)
        #print(df)  
        x, y = extract_x_y(df)
        data = make_data(x.to_numpy(), y.to_numpy())
        #データローダの設定
        vali_dataloader = DataLoader(data, batch_size=1, shuffle=False)
        for (x, t) in vali_dataloader:
            x, t = x.to(device), t.to(device)
            model.eval() #ネットワークを推論モードに
            preds = model(x)
            #print(preds)
            pre = preds
            #print(pre)
            pre = torch.argmax(pre)
            #print(pre)
        if flag["sell_signal"] == 1:
            s, a, times, flag = close_signal(now_price, flag, account_id, api, b, s, a, lot, times)
        if flag["buy_signal"] == 1:
            b, a, times, flag = close_signal(now_price, flag, account_id, api, b, s, a, lot, times)
        if pre == 0:
            b, times, flag = buy_signal(now_price, flag, account_id, api, b, s, a, lot, times)
        elif pre == 2:
            s, times, flag = sell_signal(now_price, flag, account_id, api, b, s, a, lot, times)
        
    else:
        times += 1
    time.sleep(5)


