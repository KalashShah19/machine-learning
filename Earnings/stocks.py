import pandas as pd

data = pd.read_csv("stocks.csv")

total = data[" profit"].sum()
min = data[" profit"].min()
max = data[" profit"].max()
avgPer = data[" percentage"].mean()
average = data[" profit"].sum() / data[" revenue"].sum() * 100
growth = data[' profit'].pct_change() * 100

print(f"Total Profit = {total} Rs")
print(f"Min Profit = {min} Rs")
print(f"Max Profit = {max} Rs")
print(f"Overall Average Profit = {avgPer:.2f} %")
print(f"Actual Average Profit = {average:.2f} %")
print("Growth : ")
print(growth)