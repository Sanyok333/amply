import requests
import numpy as np
import matplotlib.pyplot as plt



def get_data():
    url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
    resp = requests.get(url).json()["results"]
    return resp


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))



def solve(alpha, beta, gamma, delta):
    n = len(delta)
    A = np.zeros(n)
    B = np.zeros(n)
    x = np.zeros(n)

    A[0] = -gamma[0] / beta[0]
    B[0] = delta[0] / beta[0]

    for i in range(1, n - 1):
        den = alpha[i] * A[i - 1] + beta[i]
        A[i] = -gamma[i] / den
        B[i] = (delta[i] - alpha[i] * B[i - 1]) / den

    x[n - 1] = (delta[n - 1] - alpha[n - 1] * B[n - 2]) / (alpha[n - 1] * A[n - 2] + beta[n - 1])
    for i in range(n - 2, -1, -1):
        x[i] = A[i] * x[i + 1] + B[i]
    return x


def build_spline_full(dist, elev):
    n_nodes = len(dist)
    n = n_nodes - 1
    h = [dist[i] - dist[i - 1] for i in range(1, n_nodes)]

    alpha = np.zeros(n_nodes)
    beta = np.zeros(n_nodes)
    gamma = np.zeros(n_nodes)
    delta = np.zeros(n_nodes)

    beta[0] = 1
    for i in range(1, n):
        alpha[i] = h[i - 1]
        beta[i] = 2 * (h[i - 1] + h[i])
        gamma[i] = h[i]
        delta[i] = 3 * ((elev[i + 1] - elev[i]) / h[i] - (elev[i] - elev[i - 1]) / h[i - 1])

    alpha[n] = h[n - 1]
    beta[n] = 2 * (h[n - 1] + h[n - 1])

    c = solve(alpha, beta, gamma, delta)

    a_coeffs = [elev[i] for i in range(n)]
    b_coeffs = []
    d_coeffs = []

    for i in range(n):
        if i < n - 1:
            d_i = (c[i + 1] - c[i]) / (3 * h[i])
            b_i = (elev[i + 1] - elev[i]) / h[i] - (h[i] / 3) * (c[i + 1] + 2 * c[i])
        else:
            d_i = -c[i] / (3 * h[i])
            b_i = (elev[i + 1] - elev[i]) / h[i] - (2 / 3) * h[i] * c[i]
        b_coeffs.append(b_i)
        d_coeffs.append(d_i)

    return a_coeffs, b_coeffs, c[:-1], d_coeffs


raw_results = get_data()
all_elev = [p["elevation"] for p in raw_results]
all_coords = [(p["latitude"], p["longitude"]) for p in raw_results]


def process_route(limit):
    subset_elev = all_elev[:limit]
    subset_coords = all_coords[:limit]
    n = len(subset_elev)

    dist = [0]
    for i in range(1, n):
        d = haversine(*subset_coords[i - 1], *subset_coords[i])
        dist.append(dist[-1] + d)

    a, b, c, d = build_spline_full(dist, subset_elev)


    x_fine = np.linspace(dist[0], dist[-1], 200)
    y_fine = []
    for x_val in x_fine:
        idx = min(np.searchsorted(dist, x_val) - 1, len(a) - 1)
        idx = max(0, idx)
        dx = x_val - dist[idx]
        y_val = a[idx] + b[idx] * dx + c[idx] * (dx ** 2) + d[idx] * (dx ** 3)
        y_fine.append(y_val)

    return dist, subset_elev, x_fine, y_fine



plt.figure(figsize=(12, 8))
for count in [10, 15, 20]:
    d, e, x_f, y_f = process_route(count)
    plt.plot(x_f, y_f, label=f'Сплайн ({count} вузлів)')
    plt.scatter(d, e, s=20)

plt.title("Порівняння інтерполяції для різної кількості вузлів")
plt.xlabel("Відстань (м)")
plt.ylabel("Висота (м)")
plt.legend()
plt.grid(True)
plt.show()

d, e, x_f, y_f = process_route(21)
total_dist = d[-1]
total_ascent = sum(max(e[i] - e[i - 1], 0) for i in range(1, len(e)))
total_descent = sum(max(e[i - 1] - e[i], 0) for i in range(1, len(e)))


grad = np.gradient(y_f, x_f) * 100
eng = 80 * 9.81 * total_ascent / 1000
kcal = eng / 4.184

print(f"--- Характеристики маршруту ---")
print(f"Довжина: {total_dist:.2f} м")
print(f"Набір висоти: {total_ascent:.2f} м")
print(f"Спуск: {total_descent:.2f} м")
print(f"Макс. підйом: {np.max(grad):.2f} %")
print(f"Механічна робота: {eng:.2f} кДж ({kcal:.2f} ккал)")