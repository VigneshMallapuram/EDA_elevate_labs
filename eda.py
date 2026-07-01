import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

df = pd.read_csv("Titanic-Dataset.csv")

print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print(df.describe())

print("\nMean:")
print(df.mean(numeric_only=True))

print("\nMedian:")
print(df.median(numeric_only=True))

print("\nStandard Deviation:")
print(df.std(numeric_only=True))

df.hist(figsize=(12,10), bins=20)

plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))

sns.boxplot(y=df["Age"])

plt.title("Boxplot of Age")

plt.show()

sns.pairplot(
    df[
        ["Age",
         "Fare",
         "Pclass",
         "SibSp",
         "Parch"]
    ]
)

plt.show()

correlation = df.corr(numeric_only=True)

print(correlation)

plt.figure(figsize=(8,6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.show()

fig = px.histogram(
    df,
    x="Age",
    title="Age Distribution"
)

fig.show()

fig = px.scatter(
    df,
    x="Age",
    y="Fare",
    color="Survived",
    title="Age vs Fare"
)

fig.show()