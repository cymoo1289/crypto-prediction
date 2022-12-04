import pandas as pd
import numpy as np
'''
data = pd.read_csv("../data/extra_data/train.csv")
only_bt_data = data.loc[data['Asset_ID']==1]
print(len(data))

only_bt_data = only_bt_data.set_index('timestamp')
only_bt_data.index = pd.to_datetime(only_bt_data.index,unit='ns')
'''
only_bt_data = pd.read_csv("../data/extra_data/only_btc.csv")
only_bt_data = only_bt_data[-20000:]
only_bt_data.to_csv("../data/only_btc.csv",mode="w",index=False)
