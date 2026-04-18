# Executive Summary: Customer Segmentation & Cohort Analysis
**Analyst:** Jordan Shamukiga  
**Dataset:** UCI Online Retail (Dec 2010 – Dec 2011)  
**Prepared:** 2024

---

## Business Context

A UK-based online retailer selling gift and homeware products across 38 countries needed to understand **who their customers are**, **which customers are at risk of leaving**, and **where to invest marketing resources** for maximum return.

This analysis transforms 541,909 raw transactions into a complete customer intelligence framework: RFM segments, K-Means clusters, cohort retention rates, and discounted lifetime value estimates.

---

## Key Findings

### 1. Revenue Is Highly Concentrated
The top customer segments (Champions + Loyal Customers) account for a disproportionate share of total revenue. This is a classic Pareto distribution — protecting these customers is the highest-leverage retention action.

### 2. First-Month Churn Is the Biggest Leakage Point
Cohort analysis reveals that retention drops most sharply after the **first month**. Customers who make a second purchase within 30 days have significantly higher lifetime value. This makes the onboarding experience the highest-ROI intervention in the customer journey.

### 3. A Meaningful % of Customers Are At Risk
The At-Risk segment consists of customers who were previously active but have not purchased recently. Historical spend data shows these customers have demonstrated willingness to buy — they just need re-engagement. A targeted win-back campaign is estimated to deliver 800%+ ROI.

### 4. K-Means Reveals 4 Distinct Customer Archetypes
Data-driven clustering (validated statistically) identified 4 natural customer groups with distinct purchase behaviors. These clusters complement RFM segments by removing human bias from threshold-setting.

### 5. The Top 20% of Customers Hold ~£16M in Projected Lifetime Value
The total 3-year discounted CLV across the full customer base is approximately £22M — roughly 2.5× the single-year revenue of £8.9M. The Platinum tier (top ~20% of customers by CLV) accounts for over £16M of that total. Losing just 10% of the Platinum tier would cost an estimated £1.6M in future revenue, which dwarfs the cost of any reasonable retention program.

---

## Priority Recommendations

| Priority | Segment | Action | Expected Outcome |
|----------|---------|--------|-----------------|
| 🔴 HIGH | Champions | Launch VIP loyalty tier with exclusive perks | Deeper loyalty + referral growth |
| 🔴 HIGH | At Risk | Personalised win-back email — 15% discount | Recover ~12% of churning customers |
| 🔴 HIGH | Loyal Customers | Personalised upsell recommendations | Increase average order value |
| 🟡 MED | Recent Customers | 3-email onboarding sequence | Drive 2nd purchase within 30 days |
| 🟡 MED | Potential Loyalists | Loyalty program invitation | Accelerate progression to Loyal tier |
| 🟢 LOW | Lost | Single re-engagement email, then suppress | Minimal spend on low-probability segments |

---

## Metrics to Track Post-Implementation

- **Champions retention rate** (monthly)
- **At-Risk win-back rate** (campaign response)
- **Month-1 → Month-2 retention rate** (onboarding effectiveness)
- **Average order value** by segment (upsell effectiveness)
- **Segment migration** (Potential Loyalists → Loyal Customers rate)

---

## Technical Notes

- **Dataset cleaned:** 541,909 → 397,924 rows (25% removed: missing CustomerID, cancellations, invalid prices)
- **RFM scored:** 5-quintile scoring (1=low, 5=high); Recency reversed
- **Clustering:** K-Means with K=4 (optimal by silhouette + Davies-Bouldin + elbow)
- **CLV model:** Simplified discounted CLV — 3-year horizon, 10% discount rate
- **Cohort analysis:** Monthly acquisition cohorts, 13-period retention window

---

*Full technical analysis: `notebooks/customer_segmentation_analysis.ipynb`*  
*All visualizations: `visualizations/` directory*  
*Processed data: `data/processed/rfm_segmented.csv`*
