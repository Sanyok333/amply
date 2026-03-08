import csv
import numpy as np
import matplotlib.pyplot as plt

print("--- 1. Зчитування середньомісячних температур з CSV ---")
with open('Масив.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Month', 'Temp'])
    data_rows = [
        (1, -2), (2, 0), (3, 5), (4, 10), (5, 15), (6, 20),
        (7, 23), (8, 22), (9, 17), (10, 10), (11, 5), (12, 0),
        (13, -10), (14, 3), (15, 7), (16, 13), (17, 19), (18, 20),
        (19, 22), (20, 21), (21, 18), (22, 15), (23, 10), (24, 3)
    ]
    writer.writerows(data_rows)

x_data = []
y_data = []
with open("Масив.csv", "r", newline="") as file:
    reader = csv.DictReader(file)
    for row in reader:
        x_data.append(float(row['Month']))
        y_data.append(float(row['Temp']))
print("Дані успішно зчитано з файлу 'Масив.csv'.")

print("--- 2. Побудова полінома методом найменших квадратів (оголошення функцій) ---")
def form_matrix(x, m):
    n_pts = len(x)
    A = np.zeros((m + 1, m + 1))
    for i in range(m + 1):
        for j in range(m + 1):
            A[i, j] = sum((x[k] ** (i + j)) for k in range(n_pts))
    return A

def form_vector(x, y, m):
    n_pts = len(x)
    b = np.zeros(m + 1)
    for i in range(m + 1):
        b[i] = sum(y[k] * (x[k] ** i) for k in range(n_pts))
    return b

def gauss_solve(A_in, b_in):
    A = np.copy(A_in)
    b = np.copy(b_in)
    n = len(b)
    for k in range(n - 1):
        max_row = k + np.argmax(np.abs(A[k:n, k]))
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            b[[k, max_row]] = b[[max_row, k]]
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] = A[i, k:] - factor * A[k, k:]
            b[i] = b[i] - factor * b[k]
    x_sol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        s = sum(A[i, j] * x_sol[j] for j in range(i + 1, n))
        x_sol[i] = (b[i] - s) / A[i, i]
    return x_sol

def polynomial(x, coef):
    if isinstance(x, (list, tuple, np.ndarray)):
        y_poly = np.zeros(len(x))
        for i in range(len(coef)):
            for k in range(len(x)):
                y_poly[k] += coef[i] * (x[k] ** i)
        return y_poly
    else:
        y_val = 0
        for i in range(len(coef)):
            y_val += coef[i] * (x ** i)
        return y_val

def variance(y_true, y_approx):
    n = len(y_true)
    return sum((y_true[i] - y_approx[i]) ** 2 for i in range(n)) / (n + 1)

print("--- 3. Обчислення дисперсії для різних степенів та вибір оптимального ---")
variances = []
max_degree = 10
for m in range(1, max_degree + 1):
    A = form_matrix(x_data, m)
    b_vec = form_vector(x_data, y_data, m)
    coef = gauss_solve(A, b_vec)
    y_approx = polynomial(x_data, coef)
    var = variance(y_data, y_approx)
    variances.append(var)
    print(f"Степінь m={m}, дисперсія={var:.4f}")

optimal_m = np.argmin(variances) + 1
print(f"\nОптимальний степінь полінома (мінімальна дисперсія): {optimal_m}")

plt.figure(figsize=(8, 4))
plt.plot(range(1, max_degree + 1), variances, 'o-g')
plt.title("Залежність дисперсії від степеня многочлена")
plt.xlabel("Степінь (m)")
plt.ylabel("Дисперсія")
plt.grid(True)
plt.show()

print("--- 4. Створення графіка апроксимації та фактичних даних ---")
A_opt = form_matrix(x_data, optimal_m)
b_opt = form_vector(x_data, y_data, optimal_m)
coef_opt = gauss_solve(A_opt, b_opt)

x_dense = np.linspace(min(x_data), max(x_data), 500)
y_dense = polynomial(x_dense, coef_opt)

plt.figure(figsize=(10, 5))
plt.plot(x_data, y_data, 'ro', label='Фактичні дані')
plt.plot(x_dense, y_dense, 'b-', label=f'Апроксимація (m={optimal_m})')
plt.title(f"Апроксимація температур (оптимальний степінь m={optimal_m})")
plt.xlabel("Місяць")
plt.ylabel("Температура")
plt.legend()
plt.grid(True)
plt.show()

print("--- 5. Табулювання і побудова графіка похибки апроксимації ---")
h1 = (max(x_data) - min(x_data)) / (20 * len(x_data))
x_err = np.arange(min(x_data), max(x_data) + h1, h1)
y_true_interp = np.interp(x_err, x_data, y_data)

plt.figure(figsize=(12, 6))
for m in range(1, max_degree + 1):
    A_m = form_matrix(x_data, m)
    b_m = form_vector(x_data, y_data, m)
    coef_m = gauss_solve(A_m, b_m)
    y_poly_err = polynomial(x_err, coef_m)
    error_m = np.abs(y_true_interp - y_poly_err)
    plt.plot(x_err, error_m, label=f'm={m}')

plt.title("Похибка апроксимації |f(x) - phi(x)| для m=1...10")
plt.xlabel("Місяць")
plt.ylabel("Похибка")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

print("--- 6. Екстраполяція прогнозу температури на наступні 3 місяці ---")
x_future = [25, 26, 27]
y_future = polynomial(x_future, coef_opt)
for i in range(3):
    print(f"Прогноз на {x_future[i]} місяць: {y_future[i]:.2f} градусів")