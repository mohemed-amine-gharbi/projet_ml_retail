"""
Script de génération du dataset synthétique — projet_ml_retail
Génère projet.csv avec 52 features et problèmes de qualité intentionnels
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import string

np.random.seed(42)
random.seed(42)
N = 2000  # nombre de clients


def random_ip():
    return f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(0,255)}"


def random_date(start="2010-01-01", end="2022-12-31"):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    delta = (e - s).days
    d = s + timedelta(days=random.randint(0, delta))
    # formats inconsistants intentionnels
    fmt = random.choice(["%d/%m/%y", "%Y-%m-%d", "%m/%d/%Y"])
    return d.strftime(fmt)


# ── Features numériques ──────────────────────────────────────────────────────
CustomerID = np.random.randint(10000, 99999, N)
Recency = np.random.randint(0, 400, N)
Frequency = np.random.randint(1, 51, N)
MonetaryTotal = np.round(np.random.uniform(-500, 15000, N), 2)
MonetaryAvg = np.round(np.random.uniform(5, 500, N), 2)
MonetaryStd = np.round(np.random.uniform(0, 500, N), 2)
MonetaryMin = np.round(np.random.uniform(-5000, 5000, N), 2)
MonetaryMax = np.round(np.random.uniform(0, 10000, N), 2)
TotalQuantity = np.random.randint(-1000, 100000, N)
AvgQtyPerTrans = np.round(np.random.uniform(1, 1000, N), 2)
MinQuantity = np.random.randint(-8000, 1, N)
MaxQuantity = np.random.randint(1, 8001, N)
CustomerTenure = np.random.randint(0, 731, N)
FirstPurchase = np.random.randint(0, 731, N)
PreferredDay = np.random.randint(0, 7, N)
PreferredHour = np.random.randint(0, 24, N)
PreferredMonth = np.random.randint(1, 13, N)
WeekendRatio = np.round(np.random.uniform(0, 1, N), 4)
AvgDaysBetween = np.round(np.random.uniform(0, 365, N), 2)
UniqueProducts = np.random.randint(1, 1001, N)
UniqueDesc = np.random.randint(1, 1001, N)
AvgProdPerTrans = np.round(np.random.uniform(1, 100, N), 2)
UniqueCountries = np.random.randint(1, 6, N)
NegQtyCount = np.random.randint(0, 101, N)
ZeroPriceCount = np.random.randint(0, 51, N)
CancelledTrans = np.random.randint(0, 51, N)
ReturnRatio = np.round(np.random.uniform(0, 1, N), 4)
TotalTrans = np.random.randint(1, 10001, N)
UniqueInvoices = np.random.randint(1, 501, N)
AvgLinesPerInv = np.round(np.random.uniform(1, 100, N), 2)

# Age : 30% NaN + quelques aberrations
Age = np.where(
    np.random.rand(N) < 0.30,
    np.nan,
    np.round(np.random.uniform(18, 81, N), 1)
)

# SupportTickets : valeurs aberrantes (-1 et 999)
SupportTickets = np.where(
    np.random.rand(N) < 0.05, -1,
    np.where(np.random.rand(N) < 0.03, 999,
             np.random.randint(0, 16, N).astype(float))
)

# Satisfaction : -1, 99 comme aberrations
Satisfaction = np.where(
    np.random.rand(N) < 0.04, -1,
    np.where(np.random.rand(N) < 0.04, 99,
             np.round(np.random.uniform(0, 5, N), 1))
)

# Churn : déséquilibré ~25% churned
Churn = np.where(np.random.rand(N) < 0.25, 1, 0)

# ── Features catégorielles ───────────────────────────────────────────────────
RFMSegment = np.random.choice(["Champions", "Fidèles", "Potentiels", "Dormants"], N,
                               p=[0.2, 0.3, 0.3, 0.2])
AgeCategory = np.random.choice(["18-24", "25-34", "35-44", "45-54", "55-64", "65+", "Inconnu"], N)
SpendingCat = np.random.choice(["Low", "Medium", "High", "VIP"], N, p=[0.35, 0.35, 0.2, 0.1])
CustomerType = np.random.choice(["Hyperactif", "Régulier", "Occasionnel", "Nouveau", "Perdu"], N,
                                 p=[0.1, 0.35, 0.3, 0.15, 0.1])
FavoriteSeason = np.random.choice(["Hiver", "Printemps", "Été", "Automne"], N)
PreferredTime = np.random.choice(["Matin", "Midi", "Après-midi", "Soir", "Nuit"], N)
Region = np.random.choice(["UK", "Europe_N", "Europe_S", "Europe_E", "Europe_C", "Asie", "Autre", "Unknown"], N,
                           p=[0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05])
LoyaltyLevel = np.random.choice(["Nouveau", "Jeune", "Établi", "Ancien", "Inconnu"], N)
ChurnRisk = np.random.choice(["Faible", "Moyen", "Élevé", "Critique"], N, p=[0.4, 0.3, 0.2, 0.1])
WeekendPref = np.random.choice(["Weekend", "Semaine", "Inconnu"], N, p=[0.35, 0.55, 0.1])
BasketSize = np.random.choice(["Petit", "Moyen", "Grand", "Inconnu"], N, p=[0.3, 0.4, 0.2, 0.1])
ProdDiversity = np.random.choice(["Spécialisé", "Modéré", "Explorateur"], N, p=[0.3, 0.4, 0.3])
Gender = np.random.choice(["M", "F", "Unknown"], N, p=[0.45, 0.45, 0.1])
AccountStatus = np.random.choice(["Active", "Suspended", "Pending", "Closed"], N,
                                  p=[0.7, 0.1, 0.1, 0.1])
Country = np.random.choice(
    ["United Kingdom", "France", "Germany", "Spain", "Italy", "Netherlands",
     "Belgium", "Switzerland", "Australia", "Japan", "USA", "Canada",
     "Portugal", "Sweden", "Denmark", "Norway", "Finland", "Poland"],
    N, p=[0.4] + [0.6/17]*17)

# NewsletterSubscribed : valeur constante (à supprimer)
NewsletterSubscribed = ["Yes"] * N

# RegistrationDate : formats inconsistants
RegistrationDate = [random_date() for _ in range(N)]

# LastLoginIP
LastLoginIP = [random_ip() for _ in range(N)]

# ── Assemblage DataFrame ─────────────────────────────────────────────────────
df = pd.DataFrame({
    "CustomerID": CustomerID,
    "Recency": Recency,
    "Frequency": Frequency,
    "MonetaryTotal": MonetaryTotal,
    "MonetaryAvg": MonetaryAvg,
    "MonetaryStd": MonetaryStd,
    "MonetaryMin": MonetaryMin,
    "MonetaryMax": MonetaryMax,
    "TotalQuantity": TotalQuantity,
    "AvgQtyPerTrans": AvgQtyPerTrans,
    "MinQuantity": MinQuantity,
    "MaxQuantity": MaxQuantity,
    "CustomerTenure": CustomerTenure,
    "FirstPurchase": FirstPurchase,
    "PreferredDay": PreferredDay,
    "PreferredHour": PreferredHour,
    "PreferredMonth": PreferredMonth,
    "WeekendRatio": WeekendRatio,
    "AvgDaysBetween": AvgDaysBetween,
    "UniqueProducts": UniqueProducts,
    "UniqueDesc": UniqueDesc,
    "AvgProdPerTrans": AvgProdPerTrans,
    "UniqueCountries": UniqueCountries,
    "NegQtyCount": NegQtyCount,
    "ZeroPriceCount": ZeroPriceCount,
    "CancelledTrans": CancelledTrans,
    "ReturnRatio": ReturnRatio,
    "TotalTrans": TotalTrans,
    "UniqueInvoices": UniqueInvoices,
    "AvgLinesPerInv": AvgLinesPerInv,
    "Age": Age,
    "SupportTickets": SupportTickets,
    "Satisfaction": Satisfaction,
    "Churn": Churn,
    "RFMSegment": RFMSegment,
    "AgeCategory": AgeCategory,
    "SpendingCat": SpendingCat,
    "CustomerType": CustomerType,
    "FavoriteSeason": FavoriteSeason,
    "PreferredTime": PreferredTime,
    "Region": Region,
    "LoyaltyLevel": LoyaltyLevel,
    "ChurnRisk": ChurnRisk,
    "WeekendPref": WeekendPref,
    "BasketSize": BasketSize,
    "ProdDiversity": ProdDiversity,
    "Gender": Gender,
    "AccountStatus": AccountStatus,
    "Country": Country,
    "NewsletterSubscribed": NewsletterSubscribed,
    "RegistrationDate": RegistrationDate,
    "LastLoginIP": LastLoginIP,
})

df.to_csv("projet.csv", index=False)
print(f"✅ Dataset généré : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"   - Valeurs manquantes (Age) : {df['Age'].isna().sum()} ({df['Age'].isna().mean():.1%})")
print(f"   - Churn positif : {df['Churn'].sum()} ({df['Churn'].mean():.1%})")
print(f"   - Aberrations SupportTickets : {(df['SupportTickets'] < 0).sum() + (df['SupportTickets'] == 999).sum()}")
