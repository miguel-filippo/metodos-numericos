import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

def calcular_polinomio_quartico_interpolador(x_vals, y_vals, y_prime_vals):
    x0, x1, x2, x3, x4 = x_vals
    y0, y1, y2 = y_vals[:3]
    y_prime_3, y_prime_4 = y_prime_vals[3:5]

    # Monta a matriz A do sistema Ac = b
    A = np.array([
        [1, x0, x0**2, x0**3, x0**4],
        [1, x1, x1**2, x1**3, x1**4],
        [1, x2, x2**2, x2**3, x2**4],
        [0, 1, 2*x3, 3*x3**2, 4*x3**3],
        [0, 1, 2*x4, 3*x4**2, 4*x4**3]
    ])

    # Monta o vetor b
    b = np.array([
        y0,
        y1,
        y2,
        y_prime_3,
        y_prime_4
    ])

    try:
        # Resolve o sistema linear para encontrar os coeficientes [a, b, c, d, e]
        coeffs = np.linalg.solve(A, b)

        # Retorna um objeto polinômio
        return Polynomial(coeffs)

    except np.linalg.LinAlgError:
        print("Erro: Não foi possível resolver o sistema linear. Verifique se os pontos são distintos.")
        return None

# Função original de deflexão da viga
y_poly = Polynomial([
    0.0,              # grau 0
    -0.004,           # grau 1
    0.0004,           # grau 2
    0.0,              # grau 3
    0.000002,         # grau 4
    0.0,              # grau 5
    -0.000000011      # grau 6
])

# Derivada da função original
y_prime_poly = y_poly.deriv(1)

# Pontos medidos
x = np.array([0.0, 2.5, 5.0, 7.5, 10.0])

"""
Os dados a seguir representam os valores medidos em três pontos da viga:
Porém, como não temos os valores medidos, vamos usar o polinômio original para simular os dados.
E, ao fim, vamos comparar os resultados obtidos com os valores medidos.
"""
# Valores medidos através do polinômio original
y = [y_poly(xi) for xi in x]

# Derivadas nos pontos medidos
y_prime = [y_prime_poly(xi) for xi in x]

# Interpolação
polinomio_interpolador = calcular_polinomio_quartico_interpolador(x, y, y_prime)

print(f"Polinômio Interpolador: {polinomio_interpolador}")

if polinomio_interpolador is None:
    raise ValueError("Erro ao calcular o polinômio interpolador.")

# Calcula curvatura
curvatura = polinomio_interpolador.deriv(2)

# Calcula o máximo da curvatura
curvatura_max = minimize_scalar(
    lambda x: -curvatura(x),  # Negativo para maximizar
    bounds=(0, 10),
    method='bounded'
)

"""
Plotando os gráficos de deflexão e curvatura com escala de 0 a 10 no eixo x e -0.015 a 0.015 no eixo y.
"""
x_vals = np.linspace(0, 10, 100)
y_vals = polinomio_interpolador(x_vals)
curvatura_vals = curvatura(x_vals)

# Plotando a deflexão
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(x_vals, y_vals, label='Deflexão', color='blue')
plt.plot(x_vals, y_poly(x_vals), label='Polinômio Original', color='orange')
plt.plot(x_vals, y_prime_poly(x_vals), label='Derivada Original', color='purple')
plt.plot(x_vals, polinomio_interpolador.deriv(m = 1)(x_vals), label='Curvatura Original', color='brown')
plt.scatter(x, y, color='red', label='Medições')
plt.title('Deflexão da Viga')
plt.xlabel('Posição (m)')
plt.ylabel('Deflexão (m)')
plt.grid(False)
plt.legend()
plt.xlim(0, 10)
plt.ylim(-0.015, 0.015)

# Plotando a curvatura
plt.subplot(1, 2, 2)
plt.plot(x_vals, curvatura_vals, label='Curvatura', color='green')
plt.scatter(curvatura_max.x, curvatura(curvatura_max.x), color='black', label='Curvatura Máxima')
plt.title('Curvatura da Viga')
plt.xlabel('Posição (m)')
plt.ylabel('Curvatura (m⁻¹)')
plt.xlim(0, 10)
plt.ylim(-0.005, 0.005)
plt.yticks(np.arange(-0.005, 0.006, 0.001))
plt.legend()
plt.grid(False)

plt.savefig('lagrange_interpolation.png')
