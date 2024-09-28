
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset locally
df = pd.read_csv('titanic.csv')

# Display the first few rows
print(df.head())

# General information about the dataset
df.info()

# Summary statistics
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Plot a heatmap of missing data
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data in the Titanic Dataset')
plt.savefig('missing_data_heatmap.png')
plt.show()

# Countplot of the 'Survived' column
sns.countplot(x='Survived', data=df)
plt.title('Survival Count (0 = Not Survived, 1 = Survived)')
plt.savefig('survival_count.png')
plt.show()

# Survival rate in percentages
survival_rate = df['Survived'].value_counts(normalize=True) * 100
print(survival_rate)

# Countplot of Passenger Class
sns.countplot(x='Pclass', data=df)
plt.title('Passenger Class Distribution')
plt.savefig('pclass_distribution.png')
plt.show()

# Histogram of Age
df['Age'].plot(kind='hist', bins=20, color='blue', edgecolor='black')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.savefig('age_distribution.png')
plt.show()

# Countplot of survival by gender
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival Count by Gender')
plt.savefig('survival_by_gender.png')
plt.show()

# Countplot of survival by passenger class
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title('Survival Count by Passenger Class')
plt.savefig('survival_by_class.png')
plt.show()

# KDE plot for age distribution by survival
plt.figure(figsize=(10, 6))
sns.kdeplot(df[df['Survived'] == 1]['Age'], label='Survived', shade=True)
sns.kdeplot(df[df['Survived'] == 0]['Age'], label='Not Survived', shade=True)
plt.title('Age Distribution by Survival')
plt.xlabel('Age')
plt.legend()
plt.savefig('age_distribution_by_survival.png')
plt.show()

# Correlation matrix heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
plt.show()
