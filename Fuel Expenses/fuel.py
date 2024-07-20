def calc_fuel_expense(distance, average, price, days):
    
    if average <= 0:
        raise ValueError("Fuel efficiency must be greater than zero.")
    if distance < 0 or price < 0:
        raise ValueError("Distance and fuel cost must be non-negative.")
    
    fuel = ((distance * days) * 2) / average
    total = fuel * price
    monthly = total / 12
    daily = total / 265
    print("Distance -", distance, "km")
    print("Travel -", distance*2 , "km")
    print("Working -", days, "days" )
    print("Yearly -", distance * days * 2, "km" )
    print("Average -", average, "kmpl" )
    print("Fuel -", fuel , "Litres")
    print("Expense -", total, "rs")
    print(f"Monthly - {monthly:.2f} rs " )
    print(f"Daily - {daily:.2f} rs " )
    

days = 265
dis = 40
avg = 40 
cost = 95

fuel_expense = calc_fuel_expense(dis, avg, cost, days)