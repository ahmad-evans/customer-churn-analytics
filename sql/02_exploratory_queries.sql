-- 02_exploratory_queries.sql
-- Who is leaving, and why? Run after 01_create_tables.sql.

-- Q1. Data check: row count and how many blank total_charges were fixed
SELECT
    COUNT(*)                                            AS total_customers,
    SUM(CASE WHEN TRIM(total_charges) = '' THEN 1 ELSE 0 END) AS blank_total_charges
FROM customers_raw;

-- Q2. Overall churn rate
SELECT
    SUM(churned)                        AS churned_customers,
    ROUND(AVG(churned) * 100, 1)        AS churn_rate_pct
FROM customers_clean;

-- Q3. Churn by contract type
SELECT
    contract,
    COUNT(*)                            AS customers,
    ROUND(AVG(churned) * 100, 1)        AS churn_rate_pct
FROM customers_clean
GROUP BY contract
ORDER BY churn_rate_pct DESC;

-- Q4. Churn by tenure group (how long they've been a customer)
SELECT
    CASE
        WHEN tenure < 12 THEN '0-11 months'
        WHEN tenure < 24 THEN '12-23 months'
        WHEN tenure < 48 THEN '24-47 months'
        ELSE '48+ months'
    END                                 AS tenure_group,
    COUNT(*)                            AS customers,
    ROUND(AVG(churned) * 100, 1)        AS churn_rate_pct
FROM customers_clean
GROUP BY 1
ORDER BY MIN(tenure);

-- Q5. Churn by payment method
SELECT
    payment_method,
    COUNT(*)                            AS customers,
    ROUND(AVG(churned) * 100, 1)        AS churn_rate_pct
FROM customers_clean
GROUP BY payment_method
ORDER BY churn_rate_pct DESC;

-- Q6. Do add-on services keep people around?
SELECT
    num_addons,
    COUNT(*)                            AS customers,
    ROUND(AVG(churned) * 100, 1)        AS churn_rate_pct
FROM customers_clean
WHERE internet_service <> 'No'
GROUP BY num_addons
ORDER BY num_addons;

-- Q7. Monthly revenue lost to churn, by contract type
SELECT
    contract,
    ROUND(SUM(CASE WHEN churned = 1 THEN monthly_charges ELSE 0 END), 2) AS monthly_revenue_lost,
    ROUND(AVG(CASE WHEN churned = 1 THEN monthly_charges END), 2)        AS avg_bill_churned,
    ROUND(AVG(CASE WHEN churned = 0 THEN monthly_charges END), 2)        AS avg_bill_stayed
FROM customers_clean
GROUP BY contract
ORDER BY monthly_revenue_lost DESC;

-- Q8. Rank the riskiest customer segments (CTE + window function)
WITH segments AS (
    SELECT
        contract,
        internet_service,
        tech_support,
        COUNT(*)                        AS customers,
        ROUND(AVG(churned) * 100, 1)    AS churn_rate_pct
    FROM customers_clean
    GROUP BY contract, internet_service, tech_support
    HAVING COUNT(*) >= 100
)
SELECT
    contract,
    internet_service,
    tech_support,
    customers,
    churn_rate_pct,
    RANK() OVER (ORDER BY churn_rate_pct DESC) AS risk_rank
FROM segments
ORDER BY risk_rank
LIMIT 10;

-- Q9. Each customer's bill compared with the average for their contract type
--     (window function partitioned by contract)
SELECT
    customer_id,
    contract,
    monthly_charges,
    ROUND(AVG(monthly_charges) OVER (PARTITION BY contract), 2)                    AS contract_avg_bill,
    ROUND(monthly_charges - AVG(monthly_charges) OVER (PARTITION BY contract), 2)  AS diff_from_avg,
    churned
FROM customers_clean
ORDER BY diff_from_avg DESC
LIMIT 10;
