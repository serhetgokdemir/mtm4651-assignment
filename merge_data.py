DATA_PATH = 'data/'
import pandas as pd
import os
import gc 

def merge_and_save_data():

    print("Train transaction ve identity okunuyor...")
    train_transaction = pd.read_csv(os.path.join(DATA_PATH, 'train_transaction.csv'))
    train_identity = pd.read_csv(os.path.join(DATA_PATH, 'train_identity.csv'))
    

    print("Train verileri birleştiriliyor (Left Merge)...")
    train_merged = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
    
    
    del train_transaction, train_identity
    gc.collect()
    
    
    print(f"Train verisi kaydediliyor: {DATA_PATH}train_merged.csv")
    train_merged.to_csv(os.path.join(DATA_PATH, 'train_merged.csv'), index=False)
    
    
    print(f"Train Shape: {train_merged.shape}")
    del train_merged
    gc.collect()
    print("Train işlemi tamamlandı.\n")

    
    
    
    print("Test transaction ve identity okunuyor...")
    test_transaction = pd.read_csv(os.path.join(DATA_PATH, 'test_transaction.csv'))
    test_identity = pd.read_csv(os.path.join(DATA_PATH, 'test_identity.csv'))
    
    
    print("Test verileri birleştiriliyor (Left Merge)...")
    test_merged = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')
    
    
    del test_transaction, test_identity
    gc.collect()
    
    
    print(f"Test verisi kaydediliyor: {DATA_PATH}test_merged.csv")
    test_merged.to_csv(os.path.join(DATA_PATH, 'test_merged.csv'), index=False)
    
    print(f"Test Shape: {test_merged.shape}")
    print("Tüm işlemler başarıyla tamamlandı!")


if __name__ == "__main__":
    merge_and_save_data()