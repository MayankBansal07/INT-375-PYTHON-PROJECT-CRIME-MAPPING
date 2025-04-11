import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set styles and defaults
sns.set_style(style='whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# Load and Clean Dataset
print("\n📥 Loading and Cleaning Dataset...")
df = pd.read_excel("cleaned_python_dataset_ca.xlsx")
df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")

# Data Cleaning
df['DATE_OCC'] = pd.to_datetime(df['DATE_OCC'], errors='coerce')
df['DATE_RPTD'] = pd.to_datetime(df['DATE_RPTD'], errors='coerce')
df['TIME_OCC'] = df['TIME_OCC'].astype(str).str.zfill(4)
df['HOUR'] = df['TIME_OCC'].str[:2].astype(int)
df['YEAR'] = df['DATE_OCC'].dt.year
df['MONTH'] = df['DATE_OCC'].dt.month
df = df.dropna(subset=['LAT', 'LON'])

# Precompute cleaned group data for later plots
crime_by_month = df.groupby(['YEAR', 'MONTH']).size().reset_index(name='INCIDENTS')
crime_by_month['DATE'] = pd.to_datetime(crime_by_month[['YEAR', 'MONTH']].assign(DAY=1))
heat_data = df.groupby(['YEAR', 'MONTH']).size().unstack(fill_value=0)

area_crime = df['AREA_NAME'].value_counts().head(10)
top_crimes = df['CRM_CD_DESC'].value_counts().head(10)
weapon_usage = df['WEAPON_DESC'].value_counts().head(10)
gender_count = df['VICT_SEX'].value_counts()
ethnicity = df['VICT_DESCENT'].value_counts().head(10)
status_counts = df['STATUS_DESC'].value_counts()

top_types = df['CRM_CD_DESC'].value_counts().nlargest(5).index
status_by_type = df[df['CRM_CD_DESC'].isin(top_types)].groupby(['CRM_CD_DESC', 'STATUS_DESC']).size().unstack()
status_by_type = status_by_type.fillna(0)

output_path = r"C:\Users\mayba\OneDrive\Desktop\Python Project\cleaned2_python_dataset_ca.xlsx"
df.to_excel(output_path, index=False)


# Basic EDA
print("\n🔎 Dataset Overview")
print(df)
print("\n🔎 Head of the dataset")
print(df.head())
print("\n🔎 Tail of the dataset")
print(df.tail())
print("\n🔎 Summary Statistics")
print(df.describe())
print("\n🔎 Info")
print(df.info())
print("\n🔎 Column Names")
print(df.columns)
print("\n🔎 Shape of Dataset")
print(df.shape)
print("\n🔎 Null Values")
print(df.isnull().sum())

# Correlation & Covariance
correlation = df.corr(numeric_only=True)
print("\n📊 Correlation Matrix")
print(correlation)

covariance = df.cov(numeric_only=True)
print("\n📊 Covariance Matrix")
print(covariance)

plt.figure()
sns.heatmap(correlation, annot=True, cmap="Blues", linewidths=0.5, fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 1. Crime Distribution and Trends Over Time
print("\n📈 Answer 1: Crime Distribution and Trends Over Time")

plt.figure()
sns.lineplot(data=crime_by_month, x='DATE', y='INCIDENTS')
plt.title('Crime Incidents Over Time')
plt.xlabel('Date')
plt.ylabel('Number of Crimes')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure()
sns.histplot(df['HOUR'], bins=24, kde=True)
plt.title('Crime Frequency by Hour of the Day')
plt.xlabel('Hour')
plt.ylabel('Number of Crimes')
plt.xticks(range(0, 24))
plt.tight_layout()
plt.show()

plt.figure()
sns.heatmap(heat_data, cmap="Reds")
plt.title('Seasonal Crime Heatmap (Month vs Year)')
plt.tight_layout()
plt.show()

# 2. Geographic Crime Analysis (Crime Hotspots)
print("\n📍 Answer 2: Geographic Crime Analysis (Crime Hotspots)")

plt.figure()
sns.barplot(x=area_crime.values, y=area_crime.index)
plt.title('Top 10 High-Crime Areas')
plt.xlabel('Number of Crimes')
plt.tight_layout()
plt.show()

# 3. Crime Type Analysis
print("\n🔍 Answer 3: Crime Type Analysis")

plt.figure()
sns.barplot(x=top_crimes.values, y=top_crimes.index, palette='magma')
plt.title('Top 10 Crime Types')
plt.xlabel('Frequency')
plt.tight_layout()
plt.show()

plt.figure()
sns.barplot(x=weapon_usage.values, y=weapon_usage.index, palette='coolwarm')
plt.title('Top 10 Weapons Used')
plt.xlabel('Frequency')
plt.tight_layout()
plt.show()

# 4. Victim Demographics Breakdown
print("\n🧍 Answer 4: Victim Demographics Breakdown")

# Filter out unrealistic ages (e.g., <0 or 100+)
df = df[(df['VICT_AGE'] > 0) & (df['VICT_AGE'] < 100)]

plt.figure()
sns.histplot(df['VICT_AGE'].dropna(), bins=20, kde=True, color='purple')
plt.title('Victim Age Distribution')
plt.xlabel('Age')
plt.ylabel('Number of Victims')
plt.tight_layout()
plt.show()

plt.figure()
sns.barplot(x=gender_count.index, y=gender_count.values, palette='pastel')
plt.title('Gender Distribution of Victims')
plt.xlabel('Gender')
plt.ylabel('Number of Victims')
plt.tight_layout()
plt.show()

plt.figure()
sns.barplot(x=ethnicity.values, y=ethnicity.index, palette='BuGn_r')
plt.title('Top 10 Affected Ethnic Groups')
plt.xlabel('Number of Victims')
plt.tight_layout()
plt.show()

# 5. Crime Resolution Status Analysis
print("\n✅ Answer 5: Crime Resolution Status Analysis")

plt.figure()
plt.pie(status_counts, labels=status_counts.index, startangle=140, autopct='%1.1f%%')
plt.title('Crime Resolution Status (Donut Chart)')
plt.tight_layout()
plt.show()

status_by_type.plot(kind='bar', stacked=True, colormap='Set2', figsize=(10, 6))
plt.title('Crime Status Breakdown for Top 5 Crime Types')
plt.xlabel('Crime Type')
plt.ylabel('Number of Cases')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
