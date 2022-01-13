import pandas as pd
import glob

data_paths = glob.glob('./naver_crawling_data/*')
df = pd.DataFrame()
for data_path in data_paths:
    df_temp = pd.read_csv(data_path)
    df = pd.concat([df, df_temp])
df.reset_index(drop=True, inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

print(df)
print(df.info())
df.to_csv('./naver_crawling_data/naver_concat_data.csv', index=False)
