import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
df=pd.read_csv("telecom_kpi_balanced.csv")
def classify_network(row):
    if (row["RSRP"] >= -95 and row["SINR"] >= 20 and row["Latency"] <= 20 and row["Packet_Loss"] <= 1):
        return "Normal"
    elif (row["RSRP"] < -110 or row["SINR"] < 10 or row["Latency"] > 50 or row["Packet_Loss"] > 3):
        return "Critical"
    else:
        return "Warning"
df["Network_Status"] = df.apply(classify_network, axis=1)
print("First 5 rows")
print(df.head())
print("\nDataset Shape:",df.shape)
print("\nDataset Information") 
print(df.info())
print("Missing Values:")
print(df.isnull().sum())
df.fillna(df.mean(numeric_only=True),inplace=True)
print("\nMissing values after handling:")
print(df.isnull().sum())
print("Duplicate Records:",df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("Duplicates After:",df.duplicated().sum())
print("Dataset shape after deleting duplicates:",df.shape)
print("Encoding Target Labels")
encoder=LabelEncoder()
df["Network_Status"]=encoder.fit_transform(df["Network_Status"])
print("Label Mapping:")
for label, value in zip(encoder.classes_,encoder.transform(encoder.classes_)):
    print(f"{label}-->{value}")
X=df.drop(["Network_Status","Cell_ID","Timestamp"],axis=1)
y=df["Network_Status"]
print("\nFeature Shape:",X.shape)
print("Target Shape:",y.shape)
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)
print("\nFeature Scaling Completed")
X_train,X_temp,y_train,y_temp=train_test_split(X_scaled,y,test_size=0.30,random_state=42,stratify=y)
X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,test_size=0.50,random_state=42,stratify=y_temp)
print("\nDataset Split")
print("Training Set")
print("Features:",X_train.shape)
print("Label:",y_train.shape)
print("\nValidation Set:")
print("Features:",X_val.shape)
print("Labels:",y_val.shape)
print("\nTesting Set:")
print("Features:",X_test.shape)
print("Labels:",y_test.shape)
pd.DataFrame(X_train).to_csv("X_train.csv",index=False)
pd.DataFrame(X_val).to_csv("X_val.csv",index=False)
pd.DataFrame(X_test).to_csv("X_test.csv",index=False)
pd.DataFrame(y_train).to_csv("y_train.csv",index=False)
pd.DataFrame(y_val).to_csv("y_val.csv",index=False)
pd.DataFrame(y_test).to_csv("y_test.csv",index=False)
np.save("X_train.npy", X_train)
np.save("X_val.npy", X_val)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train.to_numpy())
np.save("y_val.npy", y_val.to_numpy())
np.save("y_test.npy", y_test.to_numpy())
print("Datasets saved successfully.")
print("\nProcessed datasets saved successfully")
print("\nDataset preparation completed successfully")