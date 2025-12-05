import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from scipy.stats import chi2_contingency
from itertools import combinations
import warnings

warnings.filterwarnings('ignore')


def test_feature_interaction(df, feature1, feature2, target='isFraud'):
    """
    Test if two features interact regarding target using Chi-Square test.

    Args:
        df (DataFrame): Input dataframe
        feature1 (str): First feature name
        feature2 (str): Second feature name
        target (str): Target variable name

    Returns:
        tuple: Chi-square statistic and p-value
    """
    contingency = pd.crosstab(
        index=[df[feature1], df[feature2]],
        columns=df[target]
    )

    chi2, p_value, dof, expected = chi2_contingency(contingency)

    print(f"Chi-Square Test: {feature1} x {feature2} -> {target}")
    print(f"Chi2 Statistic: {chi2:.2f}")
    print(f"P-value: {p_value:.6f}")

    if p_value < 0.001:
        print("STRONG interaction detected (p < 0.001)")
    elif p_value < 0.05:
        print("Significant interaction (p < 0.05)")
    else:
        print("No significant interaction")

    return chi2, p_value


def resumetable(df):
    """
    Generate comprehensive summary table of dataframe features.

    Args:
        df (DataFrame): Input dataframe

    Returns:
        DataFrame: Summary statistics table
    """
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name', 'dtypes']]
    summary['Missing'] = df.isnull().sum().values
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(
            stats.entropy(df[name].value_counts(normalize=True), base=2), 2
        )

    return summary


def plot_categorical_analysis(df, column, target='isFraud'):
    """
    Plot category distribution (pie chart) and fraud rates (stacked bar chart).

    Args:
        df (DataFrame): Input dataframe
        column (str): Column name to analyze
        target (str): Target variable name
    """
    sns.set_style('whitegrid')
    pastel_colors = sns.color_palette('pastel')

    fig, axes = plt.subplots(1, 2, figsize=(18, 6), dpi=120)

    counts = df[column].value_counts()

    axes[0].pie(
        counts,
        labels=counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=pastel_colors,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.5},
        textprops={'fontsize': 11}
    )
    axes[0].set_title(f'{column} Overall Distribution', fontsize=15, fontweight='bold')

    crosstab = pd.crosstab(df[column], df[target], normalize='index') * 100
    bar_colors = [pastel_colors[0], pastel_colors[3]]

    crosstab.plot(
        kind='bar',
        stacked=True,
        ax=axes[1],
        color=bar_colors,
        width=0.7,
        edgecolor='white'
    )

    axes[1].set_title(f'{column} Fraud vs Normal Distribution', fontsize=15, fontweight='bold')
    axes[1].set_ylabel('Rate (%)', fontsize=12)
    axes[1].set_xlabel(column, fontsize=12)
    axes[1].legend(title='Status', labels=['Normal', 'Fraud'], loc='upper right', frameon=True)
    axes[1].tick_params(axis='x', rotation=0)

    for c in axes[1].containers:
        labels = [f'{v.get_height():.1f}%' if v.get_height() > 2 else '' for v in c]
        axes[1].bar_label(c, labels=labels, label_type='center', fontsize=10, color='black', weight='bold')

    plt.tight_layout()
    plt.show()


def top_missing_cols(df, n=10, thresh=80):
    """
    Return columns with missing values percentage greater than threshold.

    Args:
        df (DataFrame): Input dataframe
        n (int): Number of top columns to return
        thresh (int): Missing percentage threshold

    Returns:
        DataFrame: Columns with missing percentages
    """
    missing_pct = (df.isnull().sum() / df.shape[0]) * 100
    missing_df = missing_pct.reset_index()
    missing_df.columns = ['col', 'missing_percent']
    missing_df = missing_df.sort_values(by=['missing_percent'], ascending=False).reset_index(drop=True)

    print(f'There are {df.isnull().any().sum()} columns with missing values.')
    print(
        f'There are {missing_df[missing_df["missing_percent"] > thresh].shape[0]} columns with missing percent > {thresh}%')

    if n:
        return missing_df.head(n)
    else:
        return missing_df


def clean_email_domains(df):
    """
    Group and clean P_emaildomain and R_emaildomain columns into major providers.

    Args:
        df (DataFrame): Input dataframe

    Returns:
        DataFrame: Dataframe with cleaned email domain columns
    """
    emails = {
        'gmail': 'google', 'gmail.com': 'google', 'googlemail.com': 'google',
        'hotmail.com': 'microsoft', 'outlook.com': 'microsoft', 'msn.com': 'microsoft',
        'live.com': 'microsoft', 'hotmail.co.uk': 'microsoft', 'hotmail.de': 'microsoft',
        'hotmail.es': 'microsoft', 'live.com.mx': 'microsoft',
        'yahoo.com': 'yahoo', 'ymail.com': 'yahoo', 'rocketmail.com': 'yahoo',
        'yahoo.com.mx': 'yahoo', 'yahoo.co.uk': 'yahoo', 'yahoo.co.jp': 'yahoo',
        'yahoo.de': 'yahoo', 'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo',
        'icloud.com': 'apple', 'me.com': 'apple', 'mac.com': 'apple',
        'aol.com': 'aol', 'aim.com': 'aol',
        'anonymous.com': 'anonymous',
        'protonmail.com': 'protonmail'
    }

    for c in ['P_emaildomain', 'R_emaildomain']:
        df[c + '_bin'] = df[c].map(emails)
        df[c + '_bin'] = df[c + '_bin'].fillna('Others')
        df.loc[df[c].isnull(), c + '_bin'] = 'Missing'

    return df


def analyze_email_match(df):
    """
    Analyze match status between purchaser and recipient email domains.

    Args:
        df (DataFrame): Input dataframe

    Returns:
        DataFrame: Dataframe with email match analysis
    """
    mask_both_exist = (df['P_emaildomain'].notnull()) & (df['R_emaildomain'].notnull())
    df['email_match'] = 'Unknown/Missing'

    df.loc[mask_both_exist, 'email_match'] = np.where(
        df.loc[mask_both_exist, 'P_emaildomain'] == df.loc[mask_both_exist, 'R_emaildomain'],
        'Match',
        'Different'
    )

    return df


def consolidate_device_info(df):
    """
    Consolidate id_30 (OS) and DeviceInfo columns into main device categories.

    Args:
        df (DataFrame): Input dataframe

    Returns:
        DataFrame: Dataframe with consolidated device information
    """
    if 'id_30' in df.columns:
        df['id_30'] = df['id_30'].astype(str).str.lower()
        df['OS_type'] = 'Others'

        df.loc[df['id_30'].str.contains('windows', na=False), 'OS_type'] = 'Windows'
        df.loc[df['id_30'].str.contains('ios', na=False), 'OS_type'] = 'iOS'
        df.loc[df['id_30'].str.contains('mac', na=False), 'OS_type'] = 'Mac'
        df.loc[df['id_30'].str.contains('android', na=False), 'OS_type'] = 'Android'
        df.loc[df['id_30'].str.contains('linux', na=False), 'OS_type'] = 'Linux'
        df.loc[df['id_30'] == 'nan', 'OS_type'] = 'Missing'

    if 'DeviceInfo' in df.columns:
        df['DeviceInfo'] = df['DeviceInfo'].astype(str).str.lower()
        df['Device_name'] = 'Others'

        df.loc[df['DeviceInfo'].str.contains('windows|trident|rv:', na=False), 'Device_name'] = 'Windows PC'
        df.loc[df['DeviceInfo'].str.contains('ios', na=False), 'Device_name'] = 'Apple Device'
        df.loc[df['DeviceInfo'].str.contains('macos|mac', na=False), 'Device_name'] = 'Mac'
        df.loc[df['DeviceInfo'].str.contains('samsung|sm-|gt-', na=False), 'Device_name'] = 'Samsung'
        df.loc[df['DeviceInfo'].str.contains('huawei|ale-|hi6210', na=False), 'Device_name'] = 'Huawei'
        df.loc[df['DeviceInfo'].str.contains('lg|lg-', na=False), 'Device_name'] = 'LG'
        df.loc[df['DeviceInfo'].str.contains('moto', na=False), 'Device_name'] = 'Motorola'
        df.loc[df['DeviceInfo'].str.contains('redmi|mi ', na=False), 'Device_name'] = 'Xiaomi'
        df.loc[df['DeviceInfo'] == 'nan', 'Device_name'] = 'Missing'

    return df


def analyze_m_columns(df):
    """
    Visualize fraud rates for M1-M9 columns (security matches).

    Args:
        df (DataFrame): Input dataframe
    """
    m_cols = [f'M{i}' for i in range(1, 10)]
    plot_data = []

    for col in m_cols:
        if col in df.columns:
            temp_df = df[[col, 'isFraud']].copy()
            temp_df[col] = temp_df[col].fillna('Missing')
            grouped = temp_df.groupby(col)['isFraud'].mean() * 100

            for val, rate in grouped.items():
                plot_data.append({
                    'Column': col,
                    'Value': val,
                    'FraudRate': rate
                })

    df_m = pd.DataFrame(plot_data)

    sns.set_style('whitegrid')
    plt.figure(figsize=(14, 8))

    sns.barplot(
        data=df_m,
        x='Column',
        y='FraudRate',
        hue='Value',
        palette='pastel',
        edgecolor='white'
    )

    plt.title('M Columns (Security Matches) Fraud Analysis', fontsize=16, fontweight='bold')
    plt.ylabel('Fraud Rate (%)', fontsize=12)
    plt.xlabel('Feature Name (M1 - M9)', fontsize=12)
    plt.axhline(df['isFraud'].mean()*100, color='black', linestyle='--', label='Overall Fraud Rate')
    plt.legend(title='Value', loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


def analyze_screen_resolution(df):
    """
    Parse screen resolution from id_33 column and analyze fraud patterns.

    Args:
        df (DataFrame): Input dataframe
    """
    df_res = df.copy()

    if 'id_33' not in df_res.columns:
        print("id_33 column not found")
        return

    resolution_split = df_res['id_33'].astype(str).str.split('x', expand=True)

    if resolution_split.shape[1] == 2:
        df_res['screen_width'] = pd.to_numeric(resolution_split[0], errors='coerce')
        df_res['screen_height'] = pd.to_numeric(resolution_split[1], errors='coerce')
        df_res['total_pixels'] = df_res['screen_width'] * df_res['screen_height']

        top_resolutions = df_res['id_33'].value_counts().head(15).index
        df_plot = df_res[df_res['id_33'].isin(top_resolutions)]

        plt.figure(figsize=(16, 8))
        res_summary = df_plot.groupby('id_33')['isFraud'].agg(['mean', 'count']).reset_index()
        res_summary['mean'] = res_summary['mean'] * 100
        res_summary = res_summary.sort_values(by='mean', ascending=False)

        ax = sns.barplot(
            data=res_summary,
            x='id_33',
            y='mean',
            palette='pastel',
            edgecolor='white'
        )

        for p, count in zip(ax.patches, res_summary['count']):
            ax.text(
                p.get_x() + p.get_width() / 2.,
                p.get_height() + 0.2,
                f"N={count}\n({p.get_height():.1f}%)",
                ha="center",
                fontsize=10,
                color='black',
                weight='bold'
            )

        plt.title('Fraud Risk by Most Common Screen Resolutions', fontsize=16)
        plt.ylabel('Fraud Rate (%)', fontsize=12)
        plt.xlabel('Screen Resolution', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
    else:
        print("id_33 format not as expected (should be WxH format)")


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Identify categorical, numerical, and high-cardinality columns based on thresholds.

    Args:
        dataframe (DataFrame): Input dataframe
        cat_th (int): Categorical threshold for unique values
        car_th (int): High cardinality threshold

    Returns:
        tuple: Lists of categorical, numerical, and high-cardinality columns
    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car


def reduce_mem_usage(df, verbose=True):
    """
    Reduce memory usage by downcasting numeric columns to appropriate data types.

    Args:
        df (DataFrame): Input dataframe
        verbose (bool): Print memory reduction information

    Returns:
        DataFrame: Memory-optimized dataframe
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Memory usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def analyze_time_categories(df):
    """
    Convert TransactionDT to hour and day categories and visualize fraud relationship.

    Args:
        df (DataFrame): Input dataframe

    Returns:
        DataFrame: Dataframe with time features
    """
    df['hour'] = (df['TransactionDT'] // 3600) % 24
    df['day_of_week'] = (df['TransactionDT'] // (3600 * 24)) % 7

    time_features = ['hour', 'day_of_week']
    titles = ['Hour of Day (0-23)', 'Day of Week (0-6)']

    sns.set_style('whitegrid')
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    for i, col in enumerate(time_features):
        ax1 = axes[i]

        total_count = df[col].value_counts().sort_index()
        fraud_rate = df.groupby(col)['isFraud'].mean() * 100
        x_indexes = total_count.index

        ax1.bar(x_indexes, total_count, color=sns.color_palette('pastel')[0],
                alpha=0.7, label='Transaction Count')
        ax1.set_ylabel('Transaction Count (Volume)', fontsize=12, color='gray')
        ax1.set_xlabel(titles[i], fontsize=12)

        ax2 = ax1.twinx()
        ax2.plot(x_indexes, fraud_rate, color='#ff7f50', marker='o',
                 linewidth=2.5, label='Fraud Rate (%)')
        ax2.set_ylabel('Fraud Rate (%)', fontsize=12, color='#ff7f50')

        ax1.set_title(f'Fraud Analysis by {titles[i]}', fontsize=15, fontweight='bold')
        ax2.grid(False)

    plt.tight_layout()
    plt.show()

    return df


def cat_binary_test(df, feature_list, target='isFraud', alpha=0.05, min_category_size=30):
    """
    Perform statistical hypothesis testing for categorical features with binary target.

    Args:
        df (DataFrame): Input dataframe
        feature_list (list): List of features to test
        target (str): Binary target variable name
        alpha (float): Significance level
        min_category_size (int): Minimum sample size per category

    Returns:
        DataFrame: Test results with statistical significance and practical importance
    """
    results = []

    for feature in feature_list:
        if feature not in df.columns:
            continue

        rates = df.groupby(feature)[target].mean()
        counts = df.groupby(feature)[target].count()

        if len(rates) < 2:
            continue

        rate_diff = rates.max() - rates.min()
        overall_rate = df[target].mean()

        contingency = pd.crosstab(df[feature], df[target])
        chi2, p_value, dof, expected = chi2_contingency(contingency)

        warning = f"Low sample size ({counts.min()})" if counts.min() < min_category_size else ""

        stat_sig = p_value < alpha
        strong_practical = rate_diff >= 0.02
        medium_practical = rate_diff >= 0.01

        if stat_sig and strong_practical:
            decision = "Strong Relation"
            keep = True
        elif stat_sig and medium_practical:
            decision = "Enough Relation"
            keep = True
        elif stat_sig:
            decision = "Poor but Meaningful"
            keep = False
        else:
            decision = "Not Related"
            keep = False

        results.append({
            'Feature': feature,
            'P_Value': round(p_value, 6),
            'Fraud_Rate_Min': round(rates.min(), 4),
            'Fraud_Rate_Max': round(rates.max(), 4),
            'Rate_Diff': round(rate_diff, 4),
            'Overall_Fraud_Rate': round(overall_rate, 4),
            'N_Categories': len(rates),
            'Min_Category_Size': int(counts.min()),
            'Decision': decision,
            'Keep': keep,
            'Warning': warning
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Rate_Diff', ascending=False).reset_index(drop=True)

    return results_df


def bivariate_comb_risk(df, feature1, feature2, target='isFraud', min_samples=30):
    """
    Analyze detailed fraud risk for two-feature combinations with heatmap visualization.

    Args:
        df (DataFrame): Input dataframe
        feature1 (str): First feature name
        feature2 (str): Second feature name
        target (str): Target variable name
        min_samples (int): Minimum sample size filter

    Returns:
        DataFrame: Top 5 riskiest combinations
    """
    df_viz = df.copy()
    df_viz[feature1] = df_viz[feature1].astype(str).fillna('Missing')
    df_viz[feature2] = df_viz[feature2].astype(str).fillna('Missing')

    group = df_viz.groupby([feature1, feature2])[target].agg(['sum', 'count', 'mean']).reset_index()
    group.columns = [feature1, feature2, 'fraud_count', 'total_count', 'fraud_rate']
    group['fraud_rate'] = group['fraud_rate'] * 100

    group = group[group['total_count'] >= min_samples].sort_values(by='fraud_rate', ascending=False)

    top_n_cat = 25
    top_f1 = df_viz[feature1].value_counts().head(top_n_cat).index
    top_f2 = df_viz[feature2].value_counts().head(top_n_cat).index

    heatmap_data = group[group[feature1].isin(top_f1) & group[feature2].isin(top_f2)]
    pivot_table = heatmap_data.pivot(index=feature1, columns=feature2, values='fraud_rate')

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn_r', center=5, ax=axes[0])
    axes[0].set_title(f'Fraud Heatmap: {feature1} vs {feature2} (Top {top_n_cat} Values)', fontsize=14)

    top_risky = group.head(10).sort_values(by='fraud_rate', ascending=True)
    labels = top_risky[feature1] + " + " + top_risky[feature2]

    bars = axes[1].barh(range(len(top_risky)), top_risky['fraud_rate'],
                        color=sns.color_palette("Reds", len(top_risky)))
    axes[1].set_yticks(range(len(top_risky)))
    axes[1].set_yticklabels(labels)
    axes[1].set_xlabel('Fraud Rate (%)')
    axes[1].set_title(f'Top 10 Riskiest Combinations ({feature1} & {feature2})', fontsize=14)

    for bar, count in zip(bars, top_risky['total_count']):
        axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f'N={count}', va='center', fontsize=9, color='black')

    plt.tight_layout()
    plt.show()

    return group.head(5)


def scan_all_bivariate_combinations(df, feature_list, target='isFraud',
                                    min_samples=50, top_n=50):
    """
    Scan all pairwise combinations of features and return top riskiest subcategories.

    Args:
        df (DataFrame): Input dataframe
        feature_list (list): List of categorical features to test
        target (str): Target variable name
        min_samples (int): Minimum sample size for combination
        top_n (int): Return top N riskiest combinations

    Returns:
        DataFrame: Top riskiest combinations with fraud rates and sample counts
    """
    df_work = df.copy()
    df_work[target] = df_work[target].astype('int32')

    for feat in feature_list:
        if feat in df_work.columns:
            df_work[feat] = df_work[feat].astype(str)

    results = []
    total_pairs = len(list(combinations(feature_list, 2)))

    print(f"Scanning {total_pairs} feature pairs...")

    for idx, (f1, f2) in enumerate(combinations(feature_list, 2), 1):
        if f1 not in df_work.columns or f2 not in df_work.columns:
            continue

        try:
            combo_stats = df_work.groupby([f1, f2], as_index=False).agg({
                target: ['sum', 'count', 'mean']
            })

            combo_stats.columns = [f1, f2, 'fraud_count', 'total_count', 'fraud_rate']
            combo_stats['fraud_rate'] = combo_stats['fraud_rate'] * 100

            combo_stats = combo_stats[combo_stats['total_count'] >= min_samples]

            if len(combo_stats) == 0:
                continue

            riskiest = combo_stats.nlargest(1, 'fraud_rate')

            if len(riskiest) > 0:
                row = riskiest.iloc[0]
                results.append({
                    'feature1': f1,
                    'feature2': f2,
                    'subcat1': str(row[f1]),
                    'subcat2': str(row[f2]),
                    'fraud_rate': row['fraud_rate'],
                    'sample_count': int(row['total_count']),
                    'fraud_count': int(row['fraud_count'])
                })
        except Exception as e:
            print(f" Error with {f1} x {f2}: {str(e)}")
            continue

        if idx % 10 == 0:
            print(f"Progress: {idx}/{total_pairs} pairs processed...")

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("No valid combinations found!")
        return results_df

    results_df = results_df.sort_values('fraud_rate', ascending=False).reset_index(drop=True)
    results_df['combination'] = (results_df['feature1'] + ' x ' + results_df['feature2'] +
                                 ': ' + results_df['subcat1'] + ' + ' + results_df['subcat2'])

    print(f"\nAnalysis complete! Found {len(results_df)} valid combinations.")
    print(f"Top fraud rate: {results_df.iloc[0]['fraud_rate']:.1f}%")

    return results_df.head(top_n)

def create_interaction_features_auto(df, top_combos_df, top_n=10, min_fraud_rate=25.0):
    """
    Otomatik olarak en riskli kombinasyonlardan yeni özellikler oluşturur.
    
    Parameters:
    -----------
    df : DataFrame
        Yeni özellikler eklenecek veri seti
    top_combos_df : DataFrame
        scan_all_bivariate_combinations() fonksiyonunun çıktısı
    top_n : int, default=10
        Kaç tane kombinasyon kullanılacak
    min_fraud_rate : float, default=15.0
        Minimum fraud oranı eşiği (%)
    
    Returns:
    --------
    df : DataFrame
        Yeni özellikler eklenmiş veri seti
        
    Example:
    --------
    >>> # Önce scan yap
    >>> top_combos = scan_all_bivariate_combinations(train_df, categorical_to_scan)
    >>> 
    >>> # Sonra otomatik birleştir
    >>> train_df = create_interaction_features_auto(train_df, top_combos, top_n=15)
    """
    
    # Filtreleme: Sadece yüksek riskli kombinasyonları al
    risky_combos = top_combos_df[
        top_combos_df['fraud_rate'] >= min_fraud_rate
    ].head(top_n)
    
    if len(risky_combos) == 0:
        print(" Hiçbir kombinasyon eşik değerini geçmiyor!")
        return df
    
    print(f"\n{'='*80}")
    print(f" Otomatik Özellik Birleştirme Başlıyor...")
    print(f"{'='*80}")
    print(f" {len(risky_combos)} kombinasyon işlenecek (fraud rate >= {min_fraud_rate}%)")
    print()
    
    created_features = []
    
    for idx, row in risky_combos.iterrows():
        feat1 = row['feature1']
        feat2 = row['feature2']
        fraud_rate = row['fraud_rate']
        
        # Yeni özellik adı
        new_feature_name = f"{feat1}_x_{feat2}"
        
        # Eğer özellikler veri setinde varsa birleştir
        if feat1 in df.columns and feat2 in df.columns:
            df[new_feature_name] = (
                df[feat1].astype(str).fillna('missing') + '_' + 
                df[feat2].astype(str).fillna('missing')
            )
            created_features.append(new_feature_name)
            
            print(f" {new_feature_name:<40} | Fraud Rate: {fraud_rate:>6.2f}%")
        else:
            print(f" {feat1} veya {feat2} bulunamadı, atlanıyor...")
    
    print()
    print(f"{'='*80}")
    print(f" Toplam {len(created_features)} yeni özellik oluşturuldu!")
    print(f"{'='*80}\n")
    
    # Oluşturulan özelliklerin listesini döndür
    df.created_interaction_features = created_features
    
    return df


if __name__ == '__main__':
    print("This file contains categorical analysis functions.")