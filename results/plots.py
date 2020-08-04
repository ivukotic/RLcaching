
import matplotlib.pyplot as plt
import pandas as pd
# import numpy as np
TB = 1024 * 1024 * 1024

df = None
# name = 'InfiniteCache_DDQN'
# name = 'InfiniteCache_LRU'
name = '100TB_LRU'
# name = '100TB_DDQN'

df = pd.read_parquet(name+'.pa')
print("data loaded:", df.shape[0])

print(df)
df['ch_files'] = df['cache hit'].cumsum()
df['CHR files'] = df['ch_files'] / df.index

df['tmp'] = df['cache hit'] * df['kB']
df['ch_data'] = df['tmp'].cumsum()
df['data delivered'] = df['kB'].cumsum()
del df['tmp']
df['CHR data'] = df['ch_data'] / df['data delivered']
df["cache size"] = df["cache size"] / TB
print(df)

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'hspace': 0.15})

ax22 = ax2.twinx()
f.suptitle(name)

ax1.plot(df["CHR files"], label='CHR files')
ax1.plot(df["CHR data"], label='CHR data')
ax1.legend()
ax1.grid(True)

ax2.plot(df["reward"].cumsum(), label='cummulative reward')
# ax22.plot(df["cache size"])
# ax22.set_ylabel('cache fill [TB]', color='b')
ax22.plot(df["reward"].rolling(5000).mean(), label='rolling reward')
ax22.set_ylabel('rolling reward', color='b')
ax2.legend()
ax2.grid(True)
# plt.tight_layout()
plt.savefig('plots/' + name + '.png')
