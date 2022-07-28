##################################################################|TRABALHO FINAL - OTIMIZAÇÃO NÃO LINEAR|#########################################################################################
# Universidade Estadual de Montes Claros - UNIMONTES
# Curso de Engenharia de Sistema - Otimização Não Linear - 5° Período
# Professor - Lenir De Abreu Júnior
#
# Equipe 01 — Métodos de Direção de Busca - Direções Aleatória
#   == GRUPO ==
#   -> Mateus Moreira Durães
#   -> Uedson Gaiek
#   -> Maria Eduarda Alves Cardoso
#   -> Thiago Emanoel Brito Santos
#   -> João Arthur Souza Santos
###################################################################################################################################################################################################
import sympy as sym
import sympy.plotting as sym_plot
import matplotlib.pyplot as plt
import numpy as np
from os import system
from random import uniform, random

# == Configurações ==
MAX = 150 # Número máximo de iterações

# Retorna um vetor gradiente de uma função
def grad(f, arg1, arg2, ptn):
    a = sym.derivative_f = f.diff(arg1) # Derivada parcial em função de x₁
    b = sym.derivative_f = f.diff(arg2) # Derivada parcial em função de x₂

    arr = [ # Vetor Gradiente
    -( (a.subs(arg1, ptn[0])).subs(arg2, ptn[1]) ),
    -( (b.subs(arg1, ptn[0])).subs(arg2, ptn[1]) )
    ]

    return arr

# == Classe para definir a condição de parada ==
class CondParada:
    def __init__(self, lmt) -> None:
        self.limite = lmt
        self.parada = False
    
    # Condição de parada baseada no Grandiente.
    def set_stop(self, func, a, b, arg):
        # Retornar falso caso apenas tenha até 5 pontos
        if len(arg) < 5:
            return False
        res = [0, 0]
        for i in range(-5, 0):
            grd = grad(func, a, b, arg[i])
            res[0] = abs(res[0]) - grd[0]
            res[1] = abs(res[1]) - grd[1]
        if (res[0] < self.limite and res[1] < self.limite):
            self.parada = True
            return True

# == Função principal ==
def otimz_rand(interval):
    # Definições de variáveis
    alpha, x1, x2 = sym.symbols('a x₁ x₂') # xi = xᵢ
    v = [4, 4] # X₀
    
    #Funções
    def func(a, b):
        return (((a-3)**2)/4) + (((b-2)**2)/9) + 13  # Função 1
       #return (a + 2*b - 7)**2 + (2*a + b - 5)**2   # Função 2

    f = func(x1, x2)
    k = 0 # N° de iterações
    pnts = [v] # Armazena Informações de todos os pontos

    # Condição de parada
    cond = CondParada(0.0001) # Tolerância de 0.01%

    print('Calculando. Por favor aguarde...')
    # Loop principal
    while (not cond.parada) and k < MAX:
        # TODO >> Mudar para variáveis aleatórias nomais ao invés de uniformes
        d = [uniform(interval[0], interval[1]), uniform(interval[0], interval[1])] # Pontos aleatórios entre n e m OBS: Mudar para normal

        p = [ # Pontos com relação de um alpha (direção)
            (v[0] + alpha * d[0]),
            (v[1] + alpha * d[1])
        ]

        temp = (f.subs(x1, p[0])).subs(x2, p[1]) # Substitui os pontos na função
        f_dif = temp.diff(alpha) # Diferencia a nova função em relação a alpha
        res = sym.solve(f_dif, alpha) # Resolve a função para f_dif = 0 para achar o valor de alpha

        if len(res) > 0:
            v = [ # Novos pontos são atribuídos
                p[0].subs(alpha, res[-1]),
                p[1].subs(alpha, res[-1])
            ]
        else: v = p
        pnts.append(v)

        # Confere a condição de parada
        cond.set_stop( f, x1, x2, pnts )
        k += 1
    
    # Resultado final
    system('cls')
    for ponto in pnts:
        print(f'{ponto}')
    print(f'Ponto ótimo X* = ({v[0]}, {v[1]}) | {k} iterações')

    # Plot do gráfico 3D
    sym_plot.plot3d(f, (x1, -100, 100), (x2, -100, 100))

    # Plot do gráfico 2D
    x, p1, p2 = [], [], []
    for x_i in range(len(pnts)) : x.append(x_i)
    for p in pnts:
        p1.append(p[0])
        p2.append(p[1])

    a = np.arange(v[0]-2, v[0]+2, 0.01)
    b = np.arange(v[1]-2, v[1]+2, 0.01)
    [X, Y] = np.meshgrid(a, b)
    Z = func(X, Y)
    fig, axis = plt.subplots(2, 1)

    axis[0].plot(x, pnts)
    axis[0].set_title('f(x₁, x₂) x Iteração')

    axis[1].contour(X, Y, Z)
    axis[1].plot(p1, p2)
    axis[1].set_title('Cruva de níveis e Deslocamento')
    plt.xlabel('x1', fontsize=10)
    plt.ylabel('x2', fontsize=10)
    plt.show()

# Roda o programa
if __name__ == '__main__':
    system('cls')
    otimz_rand([-1, 1])