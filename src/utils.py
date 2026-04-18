"""
Customer Segmentation & Cohort Analysis - Utility Functions
===========================================================
Author: Jordan Shamukiga
Project: Customer Segmentation & Cohort Analysis
Dataset: UCI Online Retail Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy import stats
import warnings
import json

warnings.filterwarnings('ignore')


SEGMENT_COLORS = {
    'Champions': '#2ecc71',
    'Loyal Customers': '#3498db',
    'Potential Loyalists': '#9b59b6',
    'Recent Customers': '#1abc9c',
    'At Risk': '#e67e22',
    'Need Attention': '#f39c12',
    'Lost': '#e74c3c',
    'Hibernating': '#95a5a6',
}

CLUSTER_PALETTE = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6']


def set_plot_style():
    """Set consistent, professional plot styling."""
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': '#f8f9fa',
        'axes.grid': True,
        'grid.color': 'white',
        'grid.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'font.family': 'DejaVu Sans',
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 100,
    })


def load_online_retail_data(filepath):
    """Load and parse the Online Retail dataset."""
    if filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        df = pd.read_excel(filepath, dtype={'CustomerID': str})
    else:
        df = pd.read_csv(filepath, encoding='ISO-8859-1', dtype={'CustomerID': str})
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns")
    print(f"   Date range: {df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}")
    return df


def clean_retail_data(df):
    """Full cleaning pipeline for the Online Retail dataset."""
    quality_report = {}
    original_rows = len(df)

    missing_customer = df['CustomerID'].isna().sum()
    df = df.dropna(subset=['CustomerID'])
    quality_report['missing_customer_id'] = int(missing_customer)

    cancelled = df['InvoiceNo'].astype(str).str.startswith('C').sum()
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    quality_report['cancelled_orders'] = int(cancelled)

    invalid_qty = (df['Quantity'] <= 0).sum()
    df = df[df['Quantity'] > 0]
    quality_report['invalid_quantity'] = int(invalid_qty)

    invalid_price = (df['UnitPrice'] <= 0).sum()
    df = df[df['UnitPrice'] > 0]
    quality_report['invalid_price'] = int(invalid_price)

    non_product_codes = ['POST', 'D', 'M', 'S', 'AMAZONFEE', 'BANK CHARGES', 'CRUK', 'DOT', 'PADS']
    mask = df['StockCode'].astype(str).str.upper().isin([c.upper() for c in non_product_codes])
    quality_report['non_product_codes'] = int(mask.sum())
    df = df[~mask]

    df = df.copy()
    df['TotalRevenue'] = df['Quantity'] * df['UnitPrice']

    quality_report['original_rows'] = int(original_rows)
    quality_report['final_rows'] = int(len(df))
    quality_report['rows_removed'] = int(original_rows - len(df))
    quality_report['pct_retained'] = round(len(df) / original_rows * 100, 1)

    print(f"Cleaning complete: {len(df):,} rows retained ({quality_report['pct_retained']}%)")
    return df, quality_report


def compute_rfm(df, snapshot_date=None):
    """Compute RFM metrics per customer."""
    if snapshot_date is None:
        snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg(
        Recency   =('InvoiceDate', lambda x: (snapshot_date - x.max()).days),
        Frequency =('InvoiceNo',   'nunique'),
        Monetary  =('TotalRevenue', 'sum')
    ).reset_index()
    print(f"RFM computed for {len(rfm):,} customers (snapshot: {snapshot_date.date()})")
    return rfm


def score_rfm(rfm, n_bins=5):
    """Assign R, F, M quintile scores (1-5)."""
    rfm = rfm.copy()
    rfm['R_Score'] = pd.qcut(rfm['Recency'], q=n_bins,
                               labels=range(n_bins, 0, -1), duplicates='drop').astype(int)
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=n_bins,
                               labels=range(1, n_bins + 1), duplicates='drop').astype(int)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'].rank(method='first'), q=n_bins,
                               labels=range(1, n_bins + 1), duplicates='drop').astype(int)
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    rfm['RFM_Total'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']
    return rfm


def assign_rfm_segment(rfm):
    """Assign business-meaningful segment labels based on R and F scores."""
    rfm = rfm.copy()
    def segment_label(row):
        r, f = row['R_Score'], row['F_Score']
        if r >= 4 and f >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3:
            return 'Loyal Customers'
        elif r >= 4 and f <= 2:
            return 'Recent Customers'
        elif r >= 3 and f >= 2:
            return 'Potential Loyalists'
        elif r <= 2 and f >= 3:
            return 'At Risk'
        elif r <= 2 and f == 2:
            return 'Need Attention'
        elif r == 1 and f <= 2:
            return 'Lost'
        else:
            return 'Hibernating'
    rfm['Segment'] = rfm.apply(segment_label, axis=1)
    return rfm


def rfm_segment_summary(rfm):
    """Return a summary DataFrame of RFM segments."""
    summary = rfm.groupby('Segment').agg(
        Customer_Count=('CustomerID', 'count'),
        Avg_Recency   =('Recency', 'mean'),
        Avg_Frequency =('Frequency', 'mean'),
        Avg_Monetary  =('Monetary', 'mean'),
        Total_Revenue =('Monetary', 'sum'),
    ).reset_index()
    summary['Pct_Customers'] = (summary['Customer_Count'] / summary['Customer_Count'].sum() * 100).round(1)
    summary['Pct_Revenue']   = (summary['Total_Revenue'] / summary['Total_Revenue'].sum() * 100).round(1)
    return summary.sort_values('Total_Revenue', ascending=False)


def prepare_features_for_clustering(rfm, log_transform=True):
    """Prepare and scale RFM features for clustering."""
    features = rfm[['Recency', 'Frequency', 'Monetary']].copy()
    if log_transform:
        features['Recency']   = np.log1p(features['Recency'])
        features['Frequency'] = np.log1p(features['Frequency'])
        features['Monetary']  = np.log1p(features['Monetary'])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    return scaled, features, scaler


def find_optimal_clusters(scaled_features, k_range=(2, 10)):
    """Compute inertia, silhouette, and Davies-Bouldin for each K."""
    results = {'k': [], 'inertia': [], 'silhouette': [], 'davies_bouldin': []}
    print("Finding optimal K...")
    for k in range(k_range[0], k_range[1] + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(scaled_features)
        results['k'].append(k)
        results['inertia'].append(km.inertia_)
        results['silhouette'].append(silhouette_score(scaled_features, labels))
        results['davies_bouldin'].append(davies_bouldin_score(scaled_features, labels))
        print(f"  K={k} | Inertia: {km.inertia_:>10.1f} | Silhouette: {results['silhouette'][-1]:.4f} | DB: {results['davies_bouldin'][-1]:.4f}")
    return results


def fit_kmeans(scaled_features, n_clusters, random_state=42):
    """Fit final K-Means model."""
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, max_iter=300)
    labels = km.fit_predict(scaled_features)
    score = silhouette_score(scaled_features, labels)
    print(f"K-Means fitted: K={n_clusters}, Silhouette={score:.4f}")
    return km, labels


def cluster_profile(rfm, cluster_col='Cluster'):
    """Generate cluster profile summary."""
    profile = rfm.groupby(cluster_col).agg(
        Count          =('CustomerID', 'count'),
        Avg_Recency    =('Recency', 'mean'),
        Avg_Frequency  =('Frequency', 'mean'),
        Avg_Monetary   =('Monetary', 'mean'),
        Median_Monetary=('Monetary', 'median'),
        Total_Revenue  =('Monetary', 'sum'),
    ).round(1)
    profile['Pct_Customers'] = (profile['Count'] / profile['Count'].sum() * 100).round(1)
    profile['Pct_Revenue']   = (profile['Total_Revenue'] / profile['Total_Revenue'].sum() * 100).round(1)
    return profile


def build_cohort_data(df):
    """Build cohort acquisition and retention data."""
    df = df.copy()
    df['OrderPeriod'] = df['InvoiceDate'].dt.to_period('M')
    df['CohortMonth'] = df.groupby('CustomerID')['InvoiceDate'].transform('min').dt.to_period('M')
    df['CohortIndex'] = (df['OrderPeriod'] - df['CohortMonth']).apply(lambda x: x.n)
    cohort_data = df.groupby(['CohortMonth', 'CohortIndex'])['CustomerID'].nunique().reset_index()
    cohort_data.columns = ['CohortMonth', 'CohortIndex', 'Customers']
    cohort_pivot = cohort_data.pivot_table(index='CohortMonth', columns='CohortIndex', values='Customers')
    cohort_size = cohort_pivot[0]
    retention = cohort_pivot.divide(cohort_size, axis=0)
    return retention, cohort_pivot, cohort_size


def compute_cohort_revenue(df):
    """Compute average revenue per customer per cohort x period."""
    df = df.copy()
    df['OrderPeriod'] = df['InvoiceDate'].dt.to_period('M')
    df['CohortMonth'] = df.groupby('CustomerID')['InvoiceDate'].transform('min').dt.to_period('M')
    df['CohortIndex'] = (df['OrderPeriod'] - df['CohortMonth']).apply(lambda x: x.n)
    rev = df.groupby(['CohortMonth', 'CohortIndex'])['TotalRevenue'].sum().reset_index()
    cust = df.groupby(['CohortMonth', 'CohortIndex'])['CustomerID'].nunique().reset_index()
    merged = rev.merge(cust, on=['CohortMonth', 'CohortIndex'])
    merged['Rev_Per_Customer'] = merged['TotalRevenue'] / merged['CustomerID']
    return merged.pivot_table(index='CohortMonth', columns='CohortIndex', values='Rev_Per_Customer')


def compute_clv(rfm, avg_lifespan_years=3, discount_rate=0.10):
    """Compute simplified CLV estimate with tier assignment."""
    rfm = rfm.copy()
    rfm['Avg_Order_Value'] = rfm['Monetary'] / rfm['Frequency']
    rfm['Annual_Revenue']  = rfm['Monetary']
    rfm['CLV_Simple']      = rfm['Annual_Revenue'] * avg_lifespan_years
    rfm['CLV_Discounted']  = rfm['Annual_Revenue'] * (
        (1 - (1 + discount_rate) ** -avg_lifespan_years) / discount_rate
    )
    q = rfm['CLV_Discounted'].quantile([0.25, 0.50, 0.75])
    rfm['CLV_Tier'] = pd.cut(
        rfm['CLV_Discounted'],
        bins=[-np.inf, q[0.25], q[0.50], q[0.75], np.inf],
        labels=['Bronze', 'Silver', 'Gold', 'Platinum']
    )
    print(f"CLV computed (lifespan={avg_lifespan_years}yr, discount={discount_rate*100:.0f}%)")
    return rfm


def describe_rfm_stats(rfm):
    """Return descriptive statistics including skewness and kurtosis."""
    stats_df = rfm[['Recency', 'Frequency', 'Monetary']].describe().round(2)
    stats_df.loc['skewness'] = rfm[['Recency', 'Frequency', 'Monetary']].skew().round(2)
    stats_df.loc['kurtosis'] = rfm[['Recency', 'Frequency', 'Monetary']].kurtosis().round(2)
    return stats_df


def kruskal_test_across_clusters(rfm, metric, cluster_col='Cluster'):
    """Non-parametric Kruskal-Wallis test across clusters."""
    groups = [g[metric].values for _, g in rfm.groupby(cluster_col)]
    stat, p = stats.kruskal(*groups)
    return {'metric': metric, 'statistic': round(stat, 4),
            'p_value': round(p, 6), 'significant': bool(p < 0.05)}


def save_figure(fig, path, dpi=150):
    """Save figure with tight layout."""
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"   Saved: {path}")


def export_metrics(metrics_dict, filepath):
    """Export key metrics to JSON."""
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, pd.Period): return str(obj)
        return str(obj)
    with open(filepath, 'w') as f:
        json.dump(metrics_dict, f, indent=2, default=convert)
    print(f"Metrics exported: {filepath}")


def print_section(title, char='=', width=60):
    """Print formatted section header."""
    print(f"\n{char * width}\n  {title}\n{char * width}")


def format_currency(value):
    return f"GBP {value:,.0f}"


def format_percent(value, decimals=1):
    return f"{value:.{decimals}f}%"
