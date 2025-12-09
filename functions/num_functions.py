import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, kstest ,ks_2samp
import warnings
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

warnings.filterwarnings('ignore')


def get_numerical_summary(df, target='isFraud', exclude_cols=None):
    """
    Generate comprehensive summary statistics for numerical features.
    
    Purpose: Quick overview of all numerical features with fraud/non-fraud comparison
    
    Args:
        df (DataFrame): Input dataframe
        target (str): Binary target variable
        exclude_cols (list): Columns to exclude (e.g., ['TransactionID'])
    
    Returns:
        DataFrame: Summary statistics with fraud rate correlations
    """
    if exclude_cols is None:
        exclude_cols = ['TransactionID', target]
    
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    summary_list = []
    
    for col in numerical_cols:
        # Basic statistics
        col_data = df[col].dropna()
        
        # Fraud vs Non-Fraud comparison
        fraud_mean = df[df[target] == 1][col].mean()
        normal_mean = df[df[target] == 0][col].mean()
        
        # Correlation with target
        corr_with_target = df[[col, target]].corr().iloc[0, 1]
        
        summary_list.append({
            'Feature': col,
            'Missing_Rate': df[col].isnull().mean() * 100,
            'Mean': col_data.mean(),
            'Std': col_data.std(),
            'Min': col_data.min(),
            'Max': col_data.max(),
            'Fraud_Mean': fraud_mean,
            'Normal_Mean': normal_mean,
            'Mean_Diff': abs(fraud_mean - normal_mean),
            'Corr_with_Target': corr_with_target,
            'Unique_Values': df[col].nunique()
        })
    
    summary_df = pd.DataFrame(summary_list)
    summary_df = summary_df.sort_values('Mean_Diff', ascending=False).reset_index(drop=True)
    
    print(f"Total numerical features: {len(numerical_cols)}")
    print(f"Features with >50% missing: {(summary_df['Missing_Rate'] > 50).sum()}")
    print(f"Features with strong correlation (|r| > 0.1): {(abs(summary_df['Corr_with_Target']) > 0.1).sum()}")
    
    return summary_df

def test_feature_discrimination(df, columns, target='isFraud', test='ks', 
                                min_samples=30, alpha=0.05):
    """
    Test if features can discriminate between fraud and normal transactions.
    
    Purpose: Identify features with statistically different distributions.
    
    Args:
        df (DataFrame): Input dataframe
        columns (list): Numerical columns to test
        target (str): Binary target variable
        test (str): 'ks' (Kolmogorov-Smirnov) or 'mw' (Mann-Whitney U)
        min_samples (int): Minimum samples required in each class
        alpha (float): Significance level (default 0.05)
    
    Returns:
        DataFrame: Test results sorted by discriminative power
    
    Output Columns:
        - Feature: Feature name
        - Test_Stat: KS statistic (D) or MW U-statistic
        - P_Value: Statistical significance
        - Significance: ***, **, *, ns (visual indicator)
        - n_fraud / n_normal: Sample sizes
        - Unique_Ratio_Fraud / Normal: Ratio of unique values (tie detection)
        - Decision: Keep/Drop recommendation
    
    Statistical Tests Explained:
        
        KS Test (Kolmogorov-Smirnov):
        - Measures maximum vertical distance between CDFs
        - D ∈ [0, 1]: 0 = identical, 1 = completely different
        - Non-parametric (no distribution assumption)
        
        Mann-Whitney U Test:
        - Tests if one distribution is stochastically greater
        - Sensitive to median differences
        - Robust to outliers
    
    Interpretation Guide:
        
        KS Statistic (D):
        - D < 0.1:  Weak discrimination (likely noise)
        - 0.1-0.3:  Moderate discrimination
        - D > 0.3:  Strong discrimination (keep feature!)
        
        P-Value:
        - p < 0.001: Very strong evidence (***) 
        - p < 0.01:  Strong evidence (**)
        - p < 0.05:  Moderate evidence (*)
        - p >= 0.05: Not significant (ns) → Drop feature
    """
    
    fraud = df[df[target] == 1]
    normal = df[df[target] == 0]
    
    results = []
    
    for col in columns:
        if col not in df.columns or col == target:
            continue
        
        # Extract values
        fraud_vals = fraud[col].dropna()
        normal_vals = normal[col].dropna()
        
        n_fraud = len(fraud_vals)
        n_normal = len(normal_vals)
        
        # Skip if insufficient samples
        if n_fraud < min_samples or n_normal < min_samples:
            continue
        
        # Statistical test
        try:
            if test == 'ks':
                stat, p = ks_2samp(fraud_vals, normal_vals)
            elif test == 'mw':
                stat, p = mannwhitneyu(fraud_vals, normal_vals, alternative='two-sided')
            else:
                raise ValueError("test must be 'ks' or 'mw'")
        except Exception as e:
            continue
        
        # Unique ratio analysis (for tie detection)
        unique_ratio_fraud = fraud_vals.nunique() / n_fraud
        unique_ratio_normal = normal_vals.nunique() / n_normal
        
        # Significance marking --> gerekli mi gerçekten bir düşünmek lazım.
        if p < 0.001:
            significance = '***'
        elif p < 0.01:
            significance = '**'
        elif p < alpha:
            significance = '*'
        else:
            significance = 'ns'
        
        # Decision logic  --> bu da gerekli olmayabilir yani okuyan kişiye bırakılması daha uygun olabilir.
        if test == 'ks':
            # KS: Higher D = better discrimination
            if p < alpha and stat > 0.1:
                decision = 'Keep'
            else:
                decision = 'Drop'
        else:
            # MW: Lower p = better
            decision = 'Keep' if p < alpha else 'Drop'
        
        results.append({
            'Feature': col,
            'Test_Stat': round(stat, 4),
            'P_Value': round(p, 6),
            'Significance': significance,
            'n_fraud': n_fraud,
            'n_normal': n_normal,
            'Unique_Ratio_Fraud': round(unique_ratio_fraud, 3),
            'Unique_Ratio_Normal': round(unique_ratio_normal, 3),
            'Decision': decision
        })
    
    # Convert to DataFrame and sort
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        print(" No features passed minimum sample threshold")
        return results_df
    
    # Sort by test statistic  --> p-val a göre de sıralanabilir.
    ascending = True if test == 'mw' else False
    results_df = results_df.sort_values('Test_Stat', ascending=ascending).reset_index(drop=True)
    
    return results_df


def plot_distribution_comparison(df, columns, target='isFraud', 
                                 plot_type='cdf', cols_per_row=3, 
                                 figsize=(5, 4), show_stats=True):
    """
    Visualize distribution differences between fraud and normal transactions.
    
    Purpose: Visual validation of statistical tests
    WHY: CDF plots work better than density for imbalanced data
    
    Args:
        df (DataFrame): Input dataframe
        columns (list): Features to plot
        target (str): Binary target
        plot_type (str): 'cdf', 'density', or 'hist'
        cols_per_row (int): Subplots per row
        figsize (tuple): Size per subplot
        show_stats (bool): Annotate with KS statistic
    """
    
    fraud = df[df[target] == 1]
    normal = df[df[target] == 0]
    
    n = len(columns)
    rows = int(np.ceil(n / cols_per_row))
    
    fig, axes = plt.subplots(rows, cols_per_row, 
                            figsize=(figsize[0] * cols_per_row, figsize[1] * rows))
    axes = np.array(axes).reshape(-1)
    
    for i, col in enumerate(columns):
        ax = axes[i]
        
        fvals = fraud[col].dropna()
        nvals = normal[col].dropna()
        
        if len(fvals) == 0 or len(nvals) == 0:
            ax.text(0.5, 0.5, f'No data\n{col}', ha='center', va='center')
            ax.axis('off')
            continue
        
        
        if plot_type == 'cdf':
            
            f_sorted = np.sort(fvals)
            n_sorted = np.sort(nvals)
            
            # Calculate empirical CDF
            f_cdf = np.arange(1, len(f_sorted) + 1) / len(f_sorted)
            n_cdf = np.arange(1, len(n_sorted) + 1) / len(n_sorted)
            
            
            ax.plot(f_sorted, f_cdf, label='Fraud', color='#ff6b6b', linewidth=2)
            ax.plot(n_sorted, n_cdf, label='Normal', color='#4ecdc4', linewidth=2)
            ax.set_ylabel('Cumulative Probability')
            
            # Add KS statistic annotation
            if show_stats:
                try:
                    ks_stat, p_val = ks_2samp(fvals, nvals)
                    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                    ax.text(0.98, 0.02, f'{sig}\nKS={ks_stat:.3f}',
                           transform=ax.transAxes, fontsize=9,
                           verticalalignment='bottom', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                except:
                    pass
        
        # kde :density
        elif plot_type == 'density':
            fvals.plot(kind='density', ax=ax, label='Fraud', 
                      color='#ff6b6b', linewidth=2)
            nvals.plot(kind='density', ax=ax, label='Normal', 
                      color='#4ecdc4', linewidth=2)
            ax.set_ylabel('Density')
        
        # histogram
        elif plot_type == 'hist':
            ax.hist(fvals, bins=40, density=True, alpha=0.5, 
                   label='Fraud', color='#ff6b6b')
            ax.hist(nvals, bins=40, density=True, alpha=0.5, 
                   label='Normal', color='#4ecdc4')
            ax.set_ylabel('Density')
        
        else:
            raise ValueError("plot_type must be 'cdf', 'density', or 'hist'")
        
        ax.set_title(col, fontweight='bold', fontsize=11)
        ax.set_xlabel(col, fontsize=9)
        ax.legend(loc='best', fontsize=9)
        ax.grid(alpha=0.3)
    
    # Remove empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_boxplot_comparison(df, columns, target='isFraud', figsize=(18, 5)):
    """
    Side-by-side boxplots for fraud vs normal transactions.
    
    Purpose: Identify outliers and median differences
    WHY: Boxplots show quartiles, median, and outliers - good for understanding data spread
    
    Args:
        df (DataFrame): Input dataframe
        columns (list): List of numerical columns to analyze
        target (str): Binary target variable
        figsize (tuple): Figure size
    
    Example:
        >>> plot_boxplot_comparison(train_df, ['TransactionAmt', 'D1', 'D2'])
    """
    n_cols = len(columns)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(figsize[0], figsize[1] * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for idx, col in enumerate(columns):
        ax = axes[idx]
        
        # Prepare data for boxplot
        plot_data = df[[col, target]].dropna()
        plot_data[target] = plot_data[target].map({0: 'Normal', 1: 'Fraud'})
        
        # Boxplot with custom colors
        sns.boxplot(data=plot_data, x=target, y=col, ax=ax, 
                   palette={'Normal': '#4ecdc4', 'Fraud': '#ff6b6b'},
                   showfliers=True)
        
        # Add median values as text
        medians = plot_data.groupby(target)[col].median()
        for xtick, median_val in zip(ax.get_xticks(), medians):
            ax.text(xtick, median_val, f'Median: {median_val:.2f}', 
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_title(f'{col} - Fraud vs Normal', fontweight='bold', fontsize=11)
        ax.set_xlabel('')
        ax.grid(axis='y', alpha=0.3)
    
    # Remove empty subplots
    for idx in range(len(columns), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.show()


def detect_outliers_iqr(df, columns, multiplier=1.5, return_indices=False):
    """
    Detect outliers using IQR (Interquartile Range) method.
    
    Purpose: Identify extreme values that might be fraud signals
    Method: IQR = Q3 - Q1
            Outlier if: value < Q1 - multiplier*IQR OR value > Q3 + multiplier*IQR
    
    Args:
        df (DataFrame): Input dataframe
        columns (list): Columns to check for outliers
        multiplier (float): IQR multiplier (1.5 = moderate, 3.0 = extreme)
        return_indices (bool): Return indices of outlier rows
    
    Returns:
        DataFrame: Outlier statistics per column
        (Optional) list: Indices of rows with outliers
    
    Example:
        >>> outlier_stats = detect_outliers_iqr(train_df, ['TransactionAmt', 'D1'])
        >>> print(outlier_stats)
    """
    outlier_stats = []
    outlier_indices = set()
    
    for col in columns:
        col_data = df[col].dropna()
        
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        # Find outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        outlier_count = len(outliers)
        outlier_pct = (outlier_count / len(df)) * 100
        
        if return_indices:
            outlier_indices.update(outliers.index.tolist())
        
        outlier_stats.append({
            'Feature': col,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound,
            'Outlier_Count': outlier_count,
            'Outlier_Pct': outlier_pct,
            'Min_Outlier': outliers.min() if len(outliers) > 0 else None,
            'Max_Outlier': outliers.max() if len(outliers) > 0 else None
        })
    
    outlier_df = pd.DataFrame(outlier_stats)
    outlier_df = outlier_df.sort_values('Outlier_Pct', ascending=False).reset_index(drop=True)
    
    print(f"Total features analyzed: {len(columns)}")
    print(f"Features with >10% outliers: {(outlier_df['Outlier_Pct'] > 10).sum()}")
    
    if return_indices:
        return outlier_df, list(outlier_indices)
    return outlier_df


def correlation_heatmap(df, columns=None, target='isFraud', method='pearson', 
                       figsize=(14, 10), top_n=None):
    """
    Generate correlation heatmap with optional target correlation sorting.
    
    Purpose: Identify highly correlated features (multicollinearity) and target relationships
    
    Args:
        df (DataFrame): Input dataframe
        columns (list): Specific columns to analyze (None = all numerical)
        target (str): Target variable to sort by
        method (str): 'pearson', 'spearman', or 'kendall'
        figsize (tuple): Figure size
        top_n (int): Show only top N features correlated with target
    
    Returns:
        DataFrame: Correlation matrix
    
    Example:
        >>> # Show top 20 features most correlated with fraud
        >>> corr_matrix = correlation_heatmap(train_df, top_n=20)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in columns:
            columns.remove(target)
    
    # Add target to analysis
    if target not in columns:
        columns = [target] + columns
    
    # Calculate correlation
    corr_matrix = df[columns].corr(method=method)
    
    # Filter top N if specified
    if top_n is not None and target in corr_matrix.columns:
        target_corr = corr_matrix[target].abs().sort_values(ascending=False)
        top_features = target_corr.head(top_n + 1).index.tolist()  # +1 for target itself
        corr_matrix = corr_matrix.loc[top_features, top_features]
    
    # Plot
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Hide upper triangle
    
    sns.heatmap(
        corr_matrix, 
        mask=mask,
        annot=True, 
        fmt='.2f', 
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        vmin=-1, vmax=1
    )
    
    plt.title(f'Correlation Heatmap ({method.capitalize()})', 
             fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.show()
    
    # Print high correlations (potential multicollinearity)
    # high_corr_pairs = []
    # for i in range(len(corr_matrix.columns)):
    #     for j in range(i+1, len(corr_matrix.columns)):
    #         if abs(corr_matrix.iloc[i, j]) > 0.7:
    #             high_corr_pairs.append({
    #                 'Feature_1': corr_matrix.columns[i],
    #                 'Feature_2': corr_matrix.columns[j],
    #                 'Correlation': corr_matrix.iloc[i, j]
    #             })
    
    # if high_corr_pairs:
    #     print("\nWarning: High correlation pairs (|r| > 0.7) - Potential multicollinearity:")
    #     for pair in high_corr_pairs[:10]:  # Show top 10
    #         print(f"  {pair['Feature_1']} <-> {pair['Feature_2']}: {pair['Correlation']:.3f}")
    
    return corr_matrix


# aşağıdaki ikisi v_col içindir !
def group_by_missing_pattern(df, columns):
    """
    Groups columns that share the exact same missing-value pattern.

    Concept:
        Two columns belong to the same group if they have NaN in exactly 
        the same rows. High-dimensional datasets like IEEE-CIS V-features 
        often contain engineered feature blocks derived from the same source, 
        resulting in identical missing masks.

    Args:
        df (DataFrame): Input dataset
        columns (list): Columns to analyze (e.g., V1–V339)

    Returns:
        dict: pattern_id → { 'columns', 'size', 'missing_rate' }
    """

    missing_patterns = {}

    for col in columns:
        pattern = tuple(df[col].isnull().values)

        if pattern not in missing_patterns:
            missing_patterns[pattern] = []

        missing_patterns[pattern].append(col)

    pattern_groups = {}

    for pattern_id, (pattern, cols) in enumerate(missing_patterns.items(), start=1):
        missing_rate = sum(pattern) / len(pattern)

        pattern_groups[pattern_id] = {
            'columns': cols,
            'size': len(cols),
            'missing_rate': missing_rate
        }

    return pattern_groups


def group_by_missing_pattern(df, columns):
    """
    Groups columns that share the exact same missing-value pattern.

    Concept:
        Two columns belong to the same group if they have NaN in exactly 
        the same rows. High-dimensional datasets like IEEE-CIS V-features 
        often contain engineered feature blocks derived from the same source, 
        resulting in identical missing masks.

    Args:
        df (DataFrame): Input dataset
        columns (list): Columns to analyze (e.g., V1–V339)

    Returns:
        dict: pattern_id → { 'columns', 'size', 'missing_rate' }
    """

    missing_patterns = {}

    for col in columns:
        pattern = tuple(df[col].isnull().values)

        if pattern not in missing_patterns:
            missing_patterns[pattern] = []

        missing_patterns[pattern].append(col)

    pattern_groups = {}

    for pattern_id, (pattern, cols) in enumerate(missing_patterns.items(), start=1):
        missing_rate = sum(pattern) / len(pattern)

        pattern_groups[pattern_id] = {
            'columns': cols,
            'size': len(cols),
            'missing_rate': missing_rate
        }

    return pattern_groups


def get_correlation_groups(df, cols, threshold=0.95, method='complete'):
    """
    Group features by correlation using hierarchical clustering.
    
    Args:
        df: DataFrame
        cols: List of columns to analyze
        threshold: Correlation threshold (default 0.95)
        method: Linkage method ('complete', 'single', 'average')
    
    Returns:
        list: List of groups, each group is a list of correlated features
    """
    
    if len(cols) <= 1:
        return [[col] for col in cols]
    
    valid_cols = [col for col in cols if col in df.columns]
    if len(valid_cols) == 0:
        return []
    
    try:
        corr_matrix = df[valid_cols].corr(method='pearson').abs()
        if corr_matrix.isnull().any().any():
            corr_matrix = corr_matrix.fillna(0)
        
        distance_matrix = 1 - corr_matrix
        condensed_dist = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_dist, method=method)
        distance_threshold = 1 - threshold
        cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
        
        groups_dict = {}
        for col, label in zip(valid_cols, cluster_labels):
            if label not in groups_dict:
                groups_dict[label] = []
            groups_dict[label].append(col)
        
        groups = list(groups_dict.values())
        
        # Print only large groups
        large_groups = [g for g in groups if len(g) > 1]
        if large_groups:
            print(f"\nCorrelation Groups (threshold={threshold}):")
            for i, group in enumerate(sorted(large_groups, key=len, reverse=True)[:5], 1):
                avg_corr = corr_matrix.loc[group, group].mean().mean()
                print(f"   {i}. Group: {len(group)} features -> {group[:5]}{'...' if len(group) > 5 else ''}")
                print(f"      Avg correlation: {avg_corr:.3f}")
        
        return groups
        
    except Exception as e:
        print(f"Error: {e}")
        return [[col] for col in valid_cols]


def select_representatives_by_ks(df, groups, target='isFraud', min_samples=30):
    """
    Select best representative from each correlation group using KS test.
    
    Args:
        df: DataFrame
        groups: Output from get_correlation_groups()
        target: Target variable (default 'isFraud')
        min_samples: Minimum samples for KS test
    
    Returns:
        tuple: (representative_list, details_df)
    """
    
    representatives = []
    selection_details = []
    
    for group_id, group in enumerate(groups, start=1):
        if len(group) == 1:
            representatives.append(group[0])
            selection_details.append({
                'Group_ID': group_id,
                'Group_Size': 1,
                'Representative': group[0],
                'Selection_Method': 'Single',
                'KS_Stat': np.nan,
                'P_Value': np.nan
            })
            continue
        
        try:
            ks_results = test_feature_discrimination(
                df, group, target=target, 
                test='ks', min_samples=min_samples
            )
            
            if not ks_results.empty:
                best_feature = ks_results.iloc[0]['Feature']
                best_ks = ks_results.iloc[0]['Test_Stat']
                best_p = ks_results.iloc[0]['P_Value']
                
                representatives.append(best_feature)
                selection_details.append({
                    'Group_ID': group_id,
                    'Group_Size': len(group),
                    'Representative': best_feature,
                    'Selection_Method': 'KS_Test',
                    'KS_Stat': best_ks,
                    'P_Value': best_p
                })
            else:
                missing_rates = df[group].isnull().mean()
                fallback = missing_rates.idxmin()
                representatives.append(fallback)
                selection_details.append({
                    'Group_ID': group_id,
                    'Group_Size': len(group),
                    'Representative': fallback,
                    'Selection_Method': 'Fallback',
                    'KS_Stat': np.nan,
                    'P_Value': np.nan
                })
        
        except Exception as e:
            fallback = group[0]
            representatives.append(fallback)
            selection_details.append({
                'Group_ID': group_id,
                'Group_Size': len(group),
                'Representative': fallback,
                'Selection_Method': 'Error',
                'KS_Stat': np.nan,
                'P_Value': np.nan
            })
    
    details_df = pd.DataFrame(selection_details)
    
    # Print selected representatives for multi-feature groups only
    multi_groups = details_df[details_df['Group_Size'] > 1]
    if not multi_groups.empty:
        print(f"\nSelected Representatives from {len(multi_groups)} groups:")
        for _, row in multi_groups.iterrows():
            if pd.notna(row['KS_Stat']):
                print(f"   Group {row['Group_ID']} ({row['Group_Size']} features): {row['Representative']} (KS={row['KS_Stat']:.3f})")
            else:
                print(f"   Group {row['Group_ID']} ({row['Group_Size']} features): {row['Representative']} (Fallback)")
    
    return representatives, details_df






def quick_numerical_eda(df, target='isFraud', top_features=10):
    """
    Run a quick comprehensive EDA on numerical features.
    
    Purpose: One-stop function for initial exploration
    
    Args:
        df (DataFrame): Input dataframe
        target (str): Target variable
        top_features (int): Number of top features to analyze in detail
    
    Example:
        >>> quick_numerical_eda(train_df, top_features=15)
    """
    print("="*80)
    print("QUICK NUMERICAL EDA REPORT")
    print("="*80)
    
    # 1. Summary statistics
    print("\n[1] Generating Summary Statistics...")
    summary = get_numerical_summary(df, target=target)
    print("\nTop 10 features by Mean Difference (Fraud vs Normal):")
    print(summary[['Feature', 'Mean_Diff', 'Corr_with_Target', 'Missing_Rate']].head(10))
    
    # 2. Top features for detailed analysis
    top_cols = summary.head(top_features)['Feature'].tolist()
    
    print(f"\n[2] Analyzing Top {top_features} Features...")
    print(f"Features: {', '.join(top_cols[:5])}...")
    
    # 3. Distribution plots
    print("\n[3] Distribution Comparison (Fraud vs Normal)...")
    analyze_numerical_distribution(df, top_cols[:6], target=target)
    
    # 4. Correlation heatmap
    print("\n[4] Correlation Analysis...")
    correlation_heatmap(df, columns=top_cols, target=target, top_n=15)
    
    print("\n" + "="*80)
    print("EDA COMPLETE! Check visualizations above.")
    print("="*80)
    
    return summary




if __name__ == '__main__':
    print("Numerical analysis utility functions loaded successfully!")
