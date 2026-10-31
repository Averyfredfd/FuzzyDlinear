import pandas as pd

for n in ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2']:
    df = pd.read_csv(f'dataset/ETT-small/{n}.csv', nrows=2)
    print(f'{n}: {list(df.columns)}  rows(nrows=2 only)')

for n in ['exchange_rate', 'weather', 'electricity', 'traffic', 'illness']:
    try:
        df = pd.read_csv(f'dataset/{n}/{n}.csv', nrows=2)
        print(f'{n}: {list(df.columns)}')
    except Exception as e:
        print(f'{n}: ERR {e}')