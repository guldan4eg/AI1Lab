from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd


df = pd.read_csv("C:\\Studing\\4sem\\AILabs\\train.csv")  # путь к файлу
scaler = MinMaxScaler()
df["Age"].fillna(df["Age"].median(), inplace=True)
df["RoomService"].fillna(df["RoomService"].median(), inplace =True)
df["FoodCourt"].fillna(df["FoodCourt"].mean(), inplace = True)
df["VRDeck"].fillna(0, inplace=True)
df = pd.get_dummies(df, columns=["HomePlanet"], drop_first=True)
df = pd.get_dummies(df, columns=["Name"], drop_first=True)
df["VRDeck"] = scaler.fit_transform(df[["VRDeck"]])
print(df.head(150))  # просмотр первых строк
df.to_csv("processed_titanic.csv", index=False)
