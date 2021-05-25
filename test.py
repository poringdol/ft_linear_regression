# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# Считывание и нормализация данных из файла

# %%
import pandas as pd

data = pd.read_csv("data.csv", sep=',', dtype={"km": float, "price": float})
m = len(data.index)

minKm = min(data["km"])
maxMinDiffKm = max(data["km"]) - minKm
minPrice = min(data["price"])
maxMinDiffPrice = max(data["price"]) - minPrice

for i, row in data.iterrows():
    row["km"] = (row["km"] - minKm) / maxMinDiffKm
    row["price"] = (row["price"] - minPrice) / maxMinDiffPrice

# %% [markdown]
# Вычисляем theta0 и theta1

# %%
from sklearn.metrics import mean_squared_error

theta0, theta1, mse = 0.0, 0.0, 0.0
learningRate = 0.1

for epoch in range(1000):
    for i,row in data.iterrows():
        arr = [theta0 + theta1 * row["km"] - row["price"] for _, row in data.iterrows()]
        theta0 -= learningRate * sum(arr) / m
        theta1 -= learningRate * sum(arr * data["km"]) / m

    estPrice = [(theta0 + theta1 * km) for km in data["km"]]
    prevMse = mse
    mse = mean_squared_error(data["price"], estPrice)
    if abs(mse - prevMse) < 0.000001:
        break

print(f"Theta0: {round(theta0, 6)}")
print(f"Theta1: {round(theta1, 6)}")
print(f"Nubers of epochs: {epoch + 1}")

# %% [markdown]
# Вычисляем стоимость для заданного пробега

# %%
mileage = int(input("Введите пробег"))

def predictPrice(km):
    normalizedMileage = (km - minKm) / maxMinDiffKm
    normalizedPrice = theta0 + theta1 * normalizedMileage
    price = normalizedPrice * maxMinDiffPrice + minPrice
    return price
price = predictPrice(mileage)
print(f"Estimate price: {round(price, 2)}")

# %% [markdown]
# График

# %%
# %matpotlib inline
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv", sep=',', dtype={"km": float, "price": float})

fig, axe = plt.subplots()
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.title("Dependence of the price of the car on the mileage")

axe.plot(data["km"], data["price"], color='b', marker='o', linewidth=0)

predictGraphX = [min(data["km"]), max(data["km"])]
predictGraphY = (predictPrice(predictGraphX[0]), predictPrice(predictGraphX[1]))
axe.plot(predictGraphX, predictGraphY)

axe.plot(mileage, price, 'or-')

plt.show()

# %% [markdown]
# Вычисление точности алгоритма

# %%
predict = [round(predictPrice(km), 2) for km in data["km"]]
data["predict price"] = predict
error = [round(abs(row["price"] - row["predict price"])) / 100 for _, row in data.iterrows()]
data["error, %"] = error
print(data)
average_error = round(sum(data["error, %"]) / len(data.index), 2)
print(f"\nAverage error: {average_error}%")
# print(f"{mean_squared_error(data['price'], data['predict price'])}")
mse = mean_squared_error(data['price'], data['predict price'])
print(f"Mean squared error: {mse}")


# %%



