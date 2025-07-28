import pandas as pd

# Preview Goal11.xlsx
def preview_excel(file_path, nrows=5):
    print(f'Previewing: {file_path}')
    df = pd.read_excel(file_path)
    print('Columns:', list(df.columns))
    print(df.head(nrows))
    print('\n---\n')

if __name__ == '__main__':
    preview_excel('data/Goal11.xlsx')
    preview_excel('data/Goal13.xlsx') 