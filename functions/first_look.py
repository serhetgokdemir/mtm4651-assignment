import pandas as pd
import numpy as np # ignore : type

def categorical_summary(df, top_n=5):
    """
    Kategorik (object veya category) değişkenler için özet bir tablo oluşturur.

    Parametreler:
    df (pd.DataFrame): Analiz edilecek veri çerçevesi.
    top_n (int): Her değişken için gösterilecek en sık N adet değer.

    Döndürür:
    pd.DataFrame: Kategorik değişkenlerin özetini içeren bir tablo.
    """
    # object veya category tipindeki sütunları seç
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    summary_list = []
    
    for col in cat_cols:
        # En sık görülen 'top_n' adet değeri ve sayılarını al
        top_values = df[col].value_counts(dropna=False).nlargest(top_n)
        
        # Yüzdelik dağılımını hesapla
        top_percentages = (df[col].value_counts(dropna=False).nlargest(top_n) / len(df) * 100).round(2)
        
        # Değerleri ve yüzdeleri birleştir
        values_and_percents = [f"{val} ({perc}%)" for val, perc in zip(top_values.index, top_percentages)]
        
        # Benzersiz değer sayısı
        unique_count = df[col].nunique(dropna=False)
        
        # Özet satırını oluştur
        summary_list.append({
            'Name': col,
            'Total Unique Values': unique_count,
            f'Top {top_n} Values (Count %)': "\n".join(values_and_percents)
        })

    return pd.DataFrame(summary_list)



def outlier_detection_summary(df, factor=1.5):
    """
    Sayısal sütunlardaki IQR metoduna göre aykırı değer sayısını raporlar.

    Parametreler:
    df (pd.DataFrame): Analiz edilecek veri çerçevesi.
    factor (float): IQR çarpanı (genellikle 1.5 kullanılır).

    Döndürür:
    pd.DataFrame: Aykırı değer raporunu içeren bir tablo.
    """
    # Sayısal (int, float) tipindeki sütunları seç
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    outlier_summary = []
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Alt ve üst aykırı değer sınırlarını hesapla
        lower_bound = Q1 - (factor * IQR)
        upper_bound = Q3 + (factor * IQR)
        
        # Sınırlara uymayan aykırı değer sayısını hesapla
        outlier_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        
        # Toplam veri sayısına göre aykırı değer oranını hesapla
        outlier_percent = round((outlier_count / df.shape[0]) * 100, 2)
        
        outlier_summary.append({
            'Name': col,
            'Total Outliers': outlier_count,
            'Outlier Percent (%)': outlier_percent,
            'Lower Bound': round(lower_bound, 2),
            'Upper Bound': round(upper_bound, 2)
        })

    # Oranı %0'dan büyük olanları sırala
    summary_df = pd.DataFrame(outlier_summary)
    return summary_df[summary_df['Outlier Percent (%)'] > 0].sort_values(by='Total Outliers', ascending=False)