import math
import numpy as np
import sympy as sp

def kf_matrices(motion_model, q_param, r_param):
    if motion_model == "RW":
        F = np.zeros((2, 2), dtype=np.float32)
        L = np.eye(2, dtype=np.float32)
        H = np.eye(2, dtype=np.float32)
    elif motion_model == "NCV":
        F = np.array([[0, 0, 1, 0],
                      [0, 0, 0, 1],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]], dtype=np.float32)
        L = np.array([[0, 0],
                        [0, 0],
                        [1, 0],
                        [0, 1]], dtype=np.float32)
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=np.float32)
    elif motion_model == "NCA":
        F = np.array([[0, 0, 1, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 1, 0],
                     [0, 0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0]], dtype=np.float32)
        L = np.array([[0, 0],
                     [0, 0],
                     [0, 0],
                     [0, 0],
                     [1, 0],
                     [0, 1]], dtype=np.float32)
        H = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0]], dtype=np.float32)        
    
    F = sp.Matrix(F)
    L = sp.Matrix(L)
    
    T, q = sp.symbols('T q')
    Fi = sp.exp(F*T).subs(T, 1)

    Q = sp.integrate((Fi*L)*q*(Fi*L).T, (T, 0, T)).subs(T, 1).subs(q, q_param)
    Q = np.array(Q, dtype=np.float32)

    R = r_param * np.eye(2, dtype=np.float32)
    Fi = np.array(Fi, dtype=np.float32)
    
    return Fi, Q, H, R  

