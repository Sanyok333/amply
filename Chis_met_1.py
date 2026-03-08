import requests
import numpy as np
import matplotlib.pyplot as plt

print("--- 1. Виконати запит до відкритого АРІ висот ---")
print("--- 2. Отримання значень широти, довготи, висоти ---")
url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
response = requests.get(url)
data = response.json()
results = data["results"]
n = len(results)

print("--- 3. Результати табуляції записати в текстовий файл ---")
with open("Масив.txt", "w", encoding="utf-8") as file:
    file.write("Latitude\tLongitude\tElevation\n")
    for p in results:
        file.write(f"{p['latitude']}\t{p['longitude']}\t{p['elevation']}\n")
print("Дані успішно записані у файл 'Масив.txt'.")

print("--- 4. Вивести кумулятивну відстань ---")


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


coords = [(p["latitude"], p["longitude"]) for p in results]
elevations = [p["elevation"] for p in results]
distances = [0]
for i in range(1, n):
    d = haversine(*coords[i - 1], *coords[i])
    distances.append(distances[-1] + d)

for i in range(n):
    print(f"Вузол {i}: Відстань = {distances[i]:.2f} м, Висота = {elevations[i]:.2f} м")

print("--- 5. Дискретний набір точок та графік кумулятивної відстані ---")
plt.figure(figsize=(10, 5))
plt.plot(distances, elevations, 'o-', label='Вхідні дані')
plt.title("Профіль висоти маршруту: Заросляк - Говерла")
plt.xlabel("Кумулятивна відстань (м)")
plt.ylabel("Висота (м)")
plt.grid(True)
plt.legend()
plt.show()

print("--- 6. Знаходження коефіцієнтів alpha, beta, gamma, delta ---")


def build_spline_system(x, y):
    n_pts = len(x)
    h = np.diff(x)
    alpha = np.zeros(n_pts)
    beta = np.zeros(n_pts)
    gamma = np.zeros(n_pts)
    delta = np.zeros(n_pts)

    beta[0] = 1.0
    gamma[0] = 0.0
    delta[0] = 0.0

    for i in range(1, n_pts - 1):
        alpha[i] = h[i - 1]
        beta[i] = 2 * (h[i - 1] + h[i])
        gamma[i] = h[i]
        delta[i] = 3 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    beta[-1] = 1.0
    alpha[-1] = 0.0
    delta[-1] = 0.0

    return h, alpha, beta, gamma, delta


h, alpha, beta, gamma, delta = build_spline_system(distances, elevations)
for i in range(n):
    print(f"i={i}: alpha={alpha[i]:.4f}, beta={beta[i]:.4f}, gamma={gamma[i]:.4f}, delta={delta[i]:.4f}")

print("--- 7. Розв'язок системи методом прогонки ---")


def thomas_algorithm(alpha, beta, gamma, delta):
    n_pts = len(delta)
    A = np.zeros(n_pts)
    B = np.zeros(n_pts)
    x = np.zeros(n_pts)

    A[0] = -gamma[0] / beta[0]
    B[0] = delta[0] / beta[0]

    for i in range(1, n_pts):
        denom = alpha[i] * A[i - 1] + beta[i]
        if i < n_pts - 1:
            A[i] = -gamma[i] / denom
        B[i] = (delta[i] - alpha[i] * B[i - 1]) / denom

    x[-1] = B[-1]
    for i in range(n_pts - 2, -1, -1):
        x[i] = A[i] * x[i + 1] + B[i]

    return x


c = thomas_algorithm(alpha, beta, gamma, delta)
for i in range(n):
    print(f"c[{i}] = {c[i]:.4f}")

print("--- 8. Обчислення коефіцієнтів C ---")
print("Коефіцієнти C вже обчислені методом прогонки вище.")

print("--- 9. Обчислення коефіцієнтів a, b, d ---")
a = np.zeros(n - 1)
b = np.zeros(n - 1)
d = np.zeros(n - 1)

for i in range(n - 1):
    a[i] = elevations[i]
    d[i] = (c[i + 1] - c[i]) / (3 * h[i])
    b[i] = (elevations[i + 1] - elevations[i]) / h[i] - (h[i] / 3) * (c[i + 1] + 2 * c[i])

for i in range(n - 1):
    print(f"Відрізок {i}: a={a[i]:.4f}, b={b[i]:.4f}, c={c[i]:.4f}, d={d[i]:.4f}")

print("--- 10-12. Графіки з різною кількістю вузлів та оцінка ---")


def eval_spline(x_eval, x_nodes, a, b, c, d):
    y_eval = np.zeros_like(x_eval)
    for i, x_val in enumerate(x_eval):
        idx = np.searchsorted(x_nodes, x_val) - 1
        if idx < 0: idx = 0
        if idx >= len(a): idx = len(a) - 1
        dx = x_val - x_nodes[idx]
        y_eval[i] = a[idx] + b[idx] * dx + c[idx] * (dx ** 2) + d[idx] * (dx ** 3)
    return y_eval


x_dense = np.linspace(min(distances), max(distances), 500)

for nodes_count in [10, 15, len(distances)]:
    idx = np.linspace(0, len(distances) - 1, nodes_count, dtype=int)
    x_sub = [distances[i] for i in idx]
    y_sub = [elevations[i] for i in idx]

    h_s, alpha_s, beta_s, gamma_s, delta_s = build_spline_system(x_sub, y_sub)
    c_s = thomas_algorithm(alpha_s, beta_s, gamma_s, delta_s)

    a_s = np.zeros(len(x_sub) - 1)
    b_s = np.zeros(len(x_sub) - 1)
    d_s = np.zeros(len(x_sub) - 1)
    for i in range(len(x_sub) - 1):
        a_s[i] = y_sub[i]
        d_s[i] = (c_s[i + 1] - c_s[i]) / (3 * h_s[i])
        b_s[i] = (y_sub[i + 1] - y_sub[i]) / h_s[i] - (h_s[i] / 3) * (c_s[i + 1] + 2 * c_s[i])

    y_dense = eval_spline(x_dense, x_sub, a_s, b_s, c_s, d_s)

    plt.figure(figsize=(10, 4))
    plt.plot(distances, elevations, 'k--', label='Оригінальні точки (справжня функція)')
    plt.plot(x_dense, y_dense, 'r-', label=f'Сплайн ({nodes_count} вузлів)')
    plt.plot(x_sub, y_sub, 'bo', label='Вузли інтерполяції')
    plt.title(f"Інтерполяція: {nodes_count} вузлів")
    plt.legend()
    plt.show()

print("--- Додатково: Характеристики маршруту ---")
print(f"Загальна довжина маршруту (м): {distances[-1]:.2f}")
total_ascent = sum(max(elevations[i] - elevations[i - 1], 0) for i in range(1, n))
print(f"Сумарний набір висоти (м): {total_ascent:.2f}")
total_descent = sum(max(elevations[i - 1] - elevations[i], 0) for i in range(1, n))
print(f"Сумарний спуск (м): {total_descent:.2f}")

y_eval_full = eval_spline(x_dense, distances, a, b, c, d)
grad_full = np.gradient(y_eval_full, x_dense) * 100
print(f"Максимальний підйом (%): {np.max(grad_full):.2f}")
print(f"Максимальний спуск (%): {np.min(grad_full):.2f}")
print(f"Середній градієнт (%): {np.mean(np.abs(grad_full)):.2f}")

mass = 80
g = 9.81
energy = mass * g * total_ascent
print(f"Механічна робота (Дж): {energy:.2f}")
print(f"Механічна робота (кДж): {energy / 1000:.2f}")
print(f"Енергія (ккал): {energy / 4184:.2f}")