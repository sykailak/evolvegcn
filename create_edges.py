import pandas as pd
import numpy as np
import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--data_path', type=argparse.FileType(mode='r'))
parser.add_argument('--save_path', type=str)
parser.add_argument('--sport', default='tennis', type=str)
args = parser.parse_args()


# Read Data
data=pd.read_csv(args.data_path)
# convert to datetime
data["placed_date"] = pd.to_datetime(data.placed_date.astype(np.int), unit='s')
# sort by placed_date
data = data.sort_values(by='placed_date',ascending=True)
data = data.reset_index(drop=True)

# Filter 02/01 ~ 02/07
data=data[data['placed_date'].dt.date >= pd.to_datetime('2020-02-01')]
data=data[data['placed_date'].dt.date <= pd.to_datetime('2020-02-07')]

# Concatenate major and league as 'majorleague'
data['majorleague']=data['major_id'].astype(str)+'_'+data['league_id'].astype(str)

# Filter sport
if args.sport == 'football':
    data = data[data.sport_id.eq(55)]
if args.sport == 'basketball':
    data = data[data.sport_id.eq(76)]
if args.sport == 'tennis':
    data = data[data.sport_id.eq(10)]
if args.sport == 'icehockey':
    data = data[data.sport_id.eq(50)]

# Print summary
print('#customers:', data['customer_id'].nunique())
print('#majorleages:', data['majorleague'].nunique())

# Picking up unique customers and mls
customers = data.customer_id.unique()
mls = data.majorleague.unique()
nodes = np.concatenate((data.customer_id.unique(), data.majorleague.unique()))

# Create dictionary with indexes
dictionary = dict(map(lambda t: (t[1], t[0]), enumerate(nodes)))

# Map to dictionary
data['customer_id'] = data['customer_id'].map(dictionary)
data['majorleague'] = data['majorleague'].map(dictionary)
data = data.reset_index(drop=True)


# Assigning timestamps

# drop minute and seconds
df = data.copy()
df['placed_date'] = df['placed_date'].dt.floor('H')

# create dictionary, mapping each hour range to indexes
start = df['placed_date'].iloc[0]
end = df['placed_date'].iloc[-1]
hour_dic = dict(zip(pd.date_range(freq='1h', start=start, end=end), range(168)))
df['time'] = df['placed_date'].map(hour_dic)

# filter
adj = df[['customer_id', 'majorleague','time']]

# add weight (1 means there exists a transaction)
adj["weight"] = 1

# rename columns
adj.columns = ['source', 'target','time','weight']

# reorder columns
adj = adj[['source', 'target','weight','time']]
adj = adj.drop_duplicates()

# duplicate the rows, switching target and source (because it's undirectional)
adj_ = adj.copy()
adj_.columns = ['target', 'source','weight','time']
adj_ = adj_[['source', 'target','weight','time']]
all_adj = adj.append(adj_, ignore_index=True)
all_adj = all_adj.drop_duplicates()

# save the final df
all_adj.to_csv(args.save_path,index=False)



