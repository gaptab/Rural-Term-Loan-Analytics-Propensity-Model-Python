# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Step 1: Generate Dummy Data for Rural Term Loan Analytics
np.random.seed(42)
num_records = 10000

# Dummy data generation
data = pd.DataFrame({
    'CustomerID': np.arange(1, num_records + 1),
    'LoanAmount': np.random.randint(50000, 500000, size=num_records),
    'GeoAccuracy': np.random.uniform(0.5, 1.0, size=num_records),
    'LeadResponseTime': np.random.randint(1, 60, size=num_records),  # in minutes
    'LeadSource': np.random.choice(['Online', 'Branch', 'Referral'], size=num_records),
    'TransactionHistory': np.random.randint(1, 50, size=num_records),  # Number of transactions
    'BehavioralScore': np.random.uniform(0, 1, size=num_records),
    'LeadConversion': np.random.choice([0, 1], size=num_records, p=[0.7, 0.3]),  # 1 = Converted
    'DisbursalAmount': np.random.randint(0, 500000, size=num_records),
    'InquiryTime': np.random.choice(pd.date_range("2024-01-01", "2024-01-31", freq='H'), size=num_records),
    'ProductType': np.random.choice(['ConsumerDurables', 'PersonalLoan'], size=num_records),
    'ChannelType': np.random.choice(['MainBranch', 'LocalSpoke'], size=num_records),
})

# Step 2: Address Verification using Geo-Location Accuracy
geo_verification_threshold = 0.8
data['NeedsPhysicalVerification'] = data['GeoAccuracy'] < geo_verification_threshold
print(f"\nPercentage of cases requiring physical verification: {data['NeedsPhysicalVerification'].mean() * 100:.2f}%")

# Step 3: Rapid Lead Response System
data['FastResponse'] = data['LeadResponseTime'] <= 15
print(f"\nPercentage of leads connected within 15 minutes: {data['FastResponse'].mean() * 100:.2f}%")

# Step 4: Propensity Model for Lead Conversion
features = ['LoanAmount', 'TransactionHistory', 'BehavioralScore', 'GeoAccuracy']
X = data[features]
y = data['LeadConversion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nLead Conversion Propensity Model:")
print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

# Calculate the impact of increased conversion rate
conversion_increase = 0.33  # 33%
monthly_disbursement = data['DisbursalAmount'].sum() * (1 + conversion_increase)
print(f"\nEstimated Monthly Disbursement after increase: {monthly_disbursement / 1e7:.2f} crores")

# Step 5: Attribution Analysis of Lead Sources
lead_source_summary = data.groupby('LeadSource').agg({
    'LeadConversion': 'mean',
    'DisbursalAmount': 'sum'
}).reset_index()
lead_source_summary['DisbursalAmount'] /= 1e7  # Convert to crores
print("\nLead Source Attribution Analysis:")
print(lead_source_summary)

# Step 6: Alternate Business Channel Analysis
channel_summary = data.groupby('ChannelType').agg({
    'LeadConversion': 'mean',
    'DisbursalAmount': 'sum'
}).reset_index()
channel_summary['DisbursalAmount'] /= 1e7  # Convert to crores
print("\nChannel Type Analysis:")
print(channel_summary)

# Step 7: Behavioral Analysis for Cross-Selling Personal Loans
cross_sell_data = data[data['ProductType'] == 'ConsumerDurables']
cross_sell_features = ['TransactionHistory', 'BehavioralScore', 'LoanAmount']
X_cross_sell = cross_sell_data[cross_sell_features]
y_cross_sell = np.random.choice([0, 1], size=len(cross_sell_data), p=[0.85, 0.15])  # Simulating conversion

model = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X_cross_sell, y_cross_sell, test_size=0.3, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nCross-Selling Analysis for Personal Loans:")
print(classification_report(y_test, y_pred))

# Calculate additional disbursal volume from cross-selling
personal_loan_increase = 0.15  # 15% contribution
additional_disbursal = monthly_disbursement * personal_loan_increase
print(f"\nAdditional disbursal volume due to cross-selling: {additional_disbursal / 1e7:.2f} crores")

# Step 8: Visualizing Impact
plt.figure(figsize=(10, 6))
plt.bar(lead_source_summary['LeadSource'], lead_source_summary['DisbursalAmount'], color='teal')
plt.title("Disbursal Amount by Lead Source (in Crores)")
plt.ylabel("Disbursal Amount (in Crores)")
plt.xlabel("Lead Source")
plt.show()

# Save the DataFrame to a CSV file
data.to_csv('rural_term_loan_analytics.csv', index=False)

print("Data has been successfully saved to 'rural_term_loan_analytics.csv'")