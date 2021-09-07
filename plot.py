import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import configparser
config = configparser.ConfigParser()
config.read('./data/account.txt')
ex_pair = config['oanda']['pair']           #対象通貨
data = pd.read_csv('./data/train_log'+'_'+ex_pair+'_'+asi+'.csv',encoding = 'UTF8')

data = data.iloc[:, 2:]

loss = data.iloc[:, 0]

acc = data.iloc[:, 1]

vali_loss = data.iloc[:, 2]

vali_acc = data.iloc[:, 3]

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

l1,l2,l3,l4 = "loss","acc","vali_loss","vali_acc" 

ax1.plot(loss, label=l1)
ax2.plot(acc, label=l2)
ax3.plot(vali_loss, label=l3)
ax4.plot(vali_acc, label=l4)

ax1.legend(loc = 'upper right') #凡例
ax2.legend(loc = 'upper right') #凡例
ax3.legend(loc = 'upper right') #凡例
ax4.legend(loc = 'upper right') #凡例
fig.tight_layout()              #レイアウトの設定
plt.show()
