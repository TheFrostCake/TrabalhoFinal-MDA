##################################################################|TRABALHO FINAL - OTIMIZAÇÃO NÃO LINEAR|#########################################################################################
# Universidade Estadual de Montes Claros - UNIMONTES
# Curso de Engenharia de Sistema - Otimização Não Linear - 5° Período
# Professor - Lenir De Abreu Júnior
#
# Equipe 01 — Métodos de Direção de Busca - Quasi-Newton
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
from random import random

# == Configurações ==
MAX = 150 # Número máximo de iterações

# Retorna um vetor gradiente de uma função
def grad(f, arg1, arg2, pnt):
    a = sym.derivative_f = f.diff(arg1) # Derivada parcial em função de x₁
    b = sym.derivative_f = f.diff(arg2) # Derivada parcial em função de x₂

    if len(pnt) > 0:
        arr = sym.Matrix([ [-((a.subs(arg1, pnt[0])).subs(arg2, pnt[1]))], [-((b.subs(arg1, pnt[0])).subs(arg2, pnt[1]))] ])
        return arr
    return sym.Matrix([[-a], [-b]])

# Retorna uma função com valores aplicados
def subs_func(f, arg1, arg2, pnt):
    return (f.subs(arg1, pnt[0])).subs(arg2, pnt[1])

# (vₖ×vₜ)/(vₜ×rₖ) - (H×rₖ×rₜ×H)/(rₜ×H×rₖ)
def get_DFP(vk, rk, h):
    vt = sym.Matrix([ [vk[0], vk[1]] ]) # Transposta de vₖ
    rt = sym.Matrix([ [rk[0], rk[1]] ]) # Transposta de rₖ

    # (vₖ×vₜ)/(vₜ×rₖ)
    p1 = (vt*rk)[0]
    p2 = (vk*vt)*(1/p1)

    # (H×rₖ×rₜ×H)/(rₜ×H×rₖ)
    p3 = (rt*h*rk)[0]
    p4 = (h*rk*rt*h)*(1/p3)

    s = p2-p4
    return s
    
# ((1 + (rₜ×H×rₖ)/(rₜ×vₖ) ) × (vₖ×vₜ)/(vₜ×rₖ)) - ((vₖ×rₜ×H + H×rₖ×vₜ)/(rₜ×vₖ))
def get_BFGS(vk, rk, h):
    vt = sym.Matrix([ [vk[0], vk[1]] ]) # Transposta de vₖ
    rt = sym.Matrix([ [rk[0], rk[1]] ]) # Transposta de rₖ

    # (1 + (rₜ×H×rₖ)/(rₜ×vₖ)
    p1 = (rt*vk)[0]
    p2 = (rt*h*rk)[0]
    p3 = 1 + (p2 * (1/p1))

    # (vₖ×vₜ)/(vₜ×rₖ)
    p4 = (vt*rk)[0]
    p5 = (vk*vt)*(1/p4)

    # ((1 + (rₜ×H×rₖ)/(rₜ×vₖ) ) × (vₖ×vₜ)/(vₜ×rₖ))
    s1 = p3 * p5

    # (vₖ×rₜ×H + H×rₖ×vₜ)/(rₜ×vₖ)
    p6 = (rt*vk)[0] 
    p7 = (vk*rt*h) + (h*rk*vt)
    s2 = p7*(1/p6)

    s = s1 - s2
    return s

def get_ck(dfp, bfgs, h):
    a = random()
    ck = ((1-a)*dfp) + (a*bfgs)
    return h+ck

# == Classe para definir a condição de parada ==
class CondParada:
    def __init__(self, lmt) -> None:
        self.limite = lmt
        self.parada = False
    
    # Parada forçada [DEBUG]
    def stop(self):
        self.parada = True

    # Condição de parada por estabilização da função
    def set_stop_stable(self, func, a, b, pnt):
        # Retorna falso caso tenha menos de 5 pontos
        if len(pnt) < 5:
            return False
        
        # Calculo do Δx
        all_values = []
        for i in pnt: all_values.append( subs_func(func, a, b, i) )
        # Δx = x₊ - x₋
        delta = max(all_values) - min(all_values)

        # Calculo do δx
        last_values = []
        for j in range(-5, 0): last_values.append(all_values[j])
        
        # δx = x₅₊ - x₅₋
        delta_x = max(last_values) - min(last_values)

        if delta_x < delta * self.limite: self.parada = True

    # Condição de parada baseada no Grandiente.
    def set_stop_grad(self, func, a, b, arg):
        # Retorna falso caso tenha menos de 3 pontos
        if len(arg) < 3:
            return False
        
        # Parada por gradiente
        grad_list = []
        for i in arg: grad_list.append( grad(func, a, b, i).norm() )
        grad_max = max(grad_list)
        # Comparar com as últimas 3 iterações 
        comp_list = []
        for j in range(-3, 0): comp_list.append(grad_list[j])
        mg = max(comp_list)

        if mg < self.limite * grad_max: self.parada = True

# == Função principal ==
def otimz_quasiNewton():
    # Definições de variáveis
    alpha, x1, x2 = sym.symbols('a x₁ x₂') # xi = xᵢ
    v = [4, 4] # X₀
    
    #Funções
    def func(a, b):
        return ((a + 2*b - 7)**2) + ((2*a+b-5)**2)   # Função Objetivo 2
        return (((a-3)**2)/4) + (((b-2)**2)/9) + 13  # Função Objetivo 1

    f = func(x1, x2)
    k = 0 # N° de iterações
    h = sym.Matrix([[1, 0], [0, 1]]) # H - indentidade
    pnts = [v] # Armazena Informações de todos os pontos

    # Condição de parada
    cond = CondParada(0.001) # Tolerância de 0.1%

    print('Calculando. Por favor aguarde...')
    # Loop principal
    while (not cond.parada) and k < MAX:
        # Direção
        g = grad(f, x1, x2, v)
        d = ((-1)*h)*g

        p = [ # Pontos com relação de um alpha (direção)
            (v[0] + alpha * d[0]),
            (v[1] + alpha * d[1])
        ]

        temp = (f.subs(x1, p[0])).subs(x2, p[1]) # Substitui os pontos na função
        f_dif = temp.diff(alpha) # Diferencia a nova função em relação a alpha
        res_alpha = sym.solve(f_dif, alpha) # Resolve a função para f_dif = 0 para achar o valor de alpha

        if len(res_alpha) > 0:
            vk = [ # Novos pontos são atribuídos
                p[0].subs(alpha, res_alpha[0]),
                p[1].subs(alpha, res_alpha[0])
            ]
            v_temp = [ v[0]-vk[0], v[1]-vk[1] ]
            v = vk
        else: v = p
        pnts.append(v)

        gk = grad(f, x1, x2, v)
        # DFP
        c_DFP = get_DFP(sym.Matrix([ [v[0]-v_temp[0]], [v[1]-v_temp[0]] ]), g-gk, h)
        # BFGS
        c_BFGS = get_BFGS(sym.Matrix([ [v[0]-v_temp[0]], [v[1]-v_temp[0]] ]), g-gk, h)
        # Ck
        h = get_ck(c_DFP, c_BFGS, h)

        # Confere a condição de parada
        cond.set_stop_stable( f, x1, x2, pnts)
       #cond.set_stop_grad( f, x1, x2, pnts )
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

    a = np.arange(v[0]-2.5, v[0]+4, 0.01)
    b = np.arange(v[1]-2.5, v[1]+4, 0.01)
    [X, Y] = np.meshgrid(a, b)
    Z = func(X, Y)
    fig, axis = plt.subplots(2, 1)

    axis[0].plot(x, pnts)
    axis[0].set_title('f(x₁, x₂) x Iteração')

    axis[1].contour(X, Y, Z)
    axis[1].plot(p1, p2)
    axis[1].set_title('Cruvas de níveis e Deslocamento')
    plt.xlabel('x1', fontsize=10)
    plt.ylabel('x2', fontsize=10)
    plt.show()

# Roda o programa
if __name__ == '__main__':
    system('cls')
    otimz_quasiNewton()