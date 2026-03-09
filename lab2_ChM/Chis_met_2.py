import numpy as np
import matplotlib.pyplot as plt
import csv

print("--- 1. Зчитування даних з CSV-файлу (Варіант 2) ---")
with open('Масив.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['RPS', 'CPU'])
    writer.writerow([50, 20])
    writer.writerow([100, 35])
    writer.writerow([200, 60])
    writer.writerow([400, 110])
    writer.writerow([800, 210])

x_data = []
y_data = []
with open("Масив.csv", "r", newline="") as file:
    reader = csv.DictReader(file)
    for row in reader:
        x_data.append(float(row['RPS']))
        y_data.append(float(row['CPU']))

print("x (RPS):", x_data)
print("y (CPU %):", y_data)

print("--- 2. Побудова таблиці розділених різниць ---")


def divided_diff(x, y):
    n = len(y)
    coef = np.zeros([n, n])
    coef[:, 0] = y
    for j in range(1, n):
        for i in range(n - j):
            coef[i][j] = (coef[i + 1][j - 1] - coef[i][j - 1]) / (x[i + j] - x[i])
    return coef


diff_table = divided_diff(x_data, y_data)
for i in range(len(x_data)):
    row_str = "\t".join([f"{diff_table[i][j]:.8f}" for j in range(len(x_data) - i)])
    print(f"Рядок {i}: {row_str}")

print("--- 3. Обчислення прогнозу для 600 RPS методами Ньютона і факторіальними многочленами ---")


def newton_interp(x_nodes, y_nodes, x_val):
    coef = divided_diff(x_nodes, y_nodes)[0, :]
    n = len(x_nodes)
    p = coef[n - 1]
    for k in range(1, n):
        p = coef[n - 1 - k] + (x_val - x_nodes[n - 1 - k]) * p
    return p


def factorial_interp(y_nodes, t_val):
    n = len(y_nodes)
    diff = np.zeros([n, n])
    diff[:, 0] = y_nodes
    for j in range(1, n):
        for i in range(n - j):
            diff[i][j] = diff[i + 1][j - 1] - diff[i][j - 1]

    forward_diffs = diff[0, :]
    result = forward_diffs[0]
    t_fact = 1
    fact = 1
    for i in range(1, n):
        t_fact *= (t_val - i + 1)
        fact *= i
        result += (forward_diffs[i] * t_fact) / fact
    return result


x_target = 600
res_newton = newton_interp(x_data, y_data, x_target)
print(f"Прогноз (Ньютон) для {x_target} RPS: {res_newton:.4f} % CPU")

h_avg = (x_data[-1] - x_data[0]) / (len(x_data) - 1)
t_target = (x_target - x_data[0]) / h_avg
res_fact = factorial_interp(y_data, t_target)
print(f"Прогноз (Факторіальні) для {x_target} RPS: {res_fact:.4f} % CPU")

print("--- 4. Побудова графіка CPU(RPS) ---")
x_plot = np.linspace(min(x_data), max(x_data), 500)
y_plot = [newton_interp(x_data, y_data, xi) for xi in x_plot]

plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, 'ro', label='Історичні дані')
plt.plot(x_plot, y_plot, 'b-', label='Поліном Ньютона')
plt.plot(x_target, res_newton, 'g*', markersize=10, label=f'Прогноз ({x_target} RPS)')
plt.title("Модель CPU = f(RPS)")
plt.xlabel("RPS (запитів за секунду)")
plt.ylabel("Використання CPU (%)")
plt.legend()
plt.grid(True)
plt.show()

print("--- 5-6. Дослідження: 5, 10, 20 вузлів та стабільність моделі (Ефект Рунге) ---")


def runge_function(x):
    return 1 / (1 + 25 * x ** 2)


a, b = -1, 1
x_dense = np.linspace(a, b, 500)
y_dense = runge_function(x_dense)

for n_nodes in [5, 10, 20]:
    x_nodes = np.linspace(a, b, n_nodes)
    y_nodes = runge_function(x_nodes)

    y_interp = [newton_interp(x_nodes, y_nodes, xi) for xi in x_dense]

    error = np.abs(y_dense - y_interp)
    max_err = np.max(error)
    print(f"Кількість вузлів: {n_nodes}. Максимальна похибка: {max_err:.6f}")

    plt.figure(figsize=(8, 4))
    plt.plot(x_dense, y_dense, 'k--', label='Справжня функція f(x)')
    plt.plot(x_dense, y_interp, 'r-', label=f'Поліном Ньютона ({n_nodes} вузлів)')
    plt.plot(x_nodes, y_nodes, 'bo', label='Вузли інтерполяції')
    plt.title(f"Ефект Рунге при {n_nodes} вузлах")
    plt.legend()
    plt.grid(True)
    plt.show()

print("\nВИСНОВОК ЩОДО СТАБІЛЬНОСТІ МОДЕЛІ:")
print("Модель показує хороші результати всередині відрізка з наявними даними.")
print("Однак, зі збільшенням кількості вузлів поліноміальна інтерполяція стає нестабільною на краях відрізка.")
print(
    "Зокрема, спостерігається ефект Рунге — різкі коливання графіка, що робить поліноми високих степенів непридатними для точного прогнозування навантаження.")