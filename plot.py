import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
import configparser
config = configparser.ConfigParser()
config.read('./data/account.txt')
ex_pair = config['oanda']['pair']           #対象通貨
asi = config['oanda']['asi']
bar = config['oanda']['bar']
lerndata = config['oanda']['lern']
data_num = config['oanda']['data']
data = pd.read_csv('./data/train_log'+'_'+ex_pair+'_'+asi+'_'+bar+'_'+lerndata+'_'+data_num+'.csv',encoding = 'UTF8')

data = data.iloc[:, 2:]

loss = data.iloc[:, 0]

acc = data.iloc[:, 1]

vali_loss = data.iloc[:, 2]

vali_acc = data.iloc[:, 3]

test_loss = data.iloc[:, 4]

test_acc = data.iloc[:, 5]

fig = plt.figure()
ax1 = fig.add_subplot(3, 2, 1)
ax2 = fig.add_subplot(3, 2, 2)
ax3 = fig.add_subplot(3, 2, 3)
ax4 = fig.add_subplot(3, 2, 4)
ax5 = fig.add_subplot(3, 2, 5)
ax6 = fig.add_subplot(3, 2, 6)

l1,l2,l3,l4,l5,l6 = "train_loss","train_acc","vali_loss","vali_acc","test_loss","test_acc" 

ax1.plot(loss, label=l1)
ax2.plot(acc, label=l2)
ax3.plot(vali_loss, label=l3)
ax4.plot(vali_acc, label=l4)
ax5.plot(test_loss, label=l5)
ax6.plot(test_acc, label=l6)

ax1.legend(loc = 'upper right') #凡例
ax2.legend(loc = 'upper right') #凡例
ax3.legend(loc = 'upper right') #凡例
ax4.legend(loc = 'upper right') #凡例
ax5.legend(loc = 'upper right') #凡例
ax6.legend(loc = 'upper right') #凡例
fig.tight_layout()              #レイアウトの設定
plt.show()
