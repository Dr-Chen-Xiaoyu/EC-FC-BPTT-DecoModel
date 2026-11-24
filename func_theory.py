'''Linearized theoretical EC and FC of the reduced Wong--Wang Deco model.'''
from pathlib import Path
import sys

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import enable_x64
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_lyapunov
from scipy.optimize import root

sys.path.append(str(Path(__file__).resolve().parents[2]))
from func_brainpy_model import get_AbbottChance


def check_model_parameters(struc_conn_matrix, G, w, I, tau_S, gamma, J_N,
                      S=0.0, sigma=0.0, delta=1.0):
    '''统一检查形状与有限值，将所有脑区参数转换为 (N,)，保留 delta=None。'''
    struc_conn_matrix = np.asarray(struc_conn_matrix, dtype=float)
    if (struc_conn_matrix.ndim != 2 or not struc_conn_matrix.shape[0]
            or struc_conn_matrix.shape[0] != struc_conn_matrix.shape[1]
            or not np.isfinite(struc_conn_matrix).all()):
        raise ValueError('struc_conn_matrix must be a finite, nonempty square matrix.')
    N = struc_conn_matrix.shape[0]

    def node_array(x):
        x = np.broadcast_to(np.asarray(x, dtype=float), (N,))
        if not np.isfinite(x).all():
            raise ValueError('Node parameters must be finite.')
        return x

    w, I, tau_S, S, sigma = [node_array(x) for x in (w, I, tau_S, S, sigma)]
    delta = None if delta is None else node_array(delta)
    G = float(np.asarray(G).reshape(()))
    if not np.isfinite([G, gamma, J_N]).all() or np.any(tau_S <= 0):
        raise ValueError('Parameters must be finite and tau_S must be positive.')
    parameters = struc_conn_matrix, G, w, I, tau_S, gamma, J_N
    return parameters, S, sigma, delta


@jax.jit
def DecoDrift(S, struc_conn_matrix, G, w, I, tau_S, gamma, J_N):
    '''连续时间 Deco 动力学的 drift；J_N 为突触耦合常数，保留背景 I，无额外刺激或噪声。'''
    x = J_N * (w * S + jnp.abs(G) * (S @ struc_conn_matrix)) + I
    H = jnp.asarray(get_AbbottChance(method='custom_jvp')(x))
    return -S / tau_S + gamma * (1-S) * H


DecoDrift_jacobian = jax.jit(jax.jacfwd(DecoDrift, argnums=0))


def get_DecoDrift_jacobian(struc_conn_matrix, G, w, I, S, tau_S=0.1, gamma=0.641, J_N=0.2609):
    '''通过 jax.jacfwd 计算指定状态 S 处的 Jacobian；行接收、列源节点。'''
    parameters, S, _, _ = check_model_parameters(struc_conn_matrix, G, w, I, tau_S, gamma, J_N, S=S)
    with enable_x64():
        return np.asarray(DecoDrift_jacobian(S, *parameters))


def get_S_star(struc_conn_matrix, G, w, I, S=0.0, tau_S=0.1, gamma=0.641, J_N=0.2609,
               duration=2.0, tol=1e-10):
    '''
    无额外刺激、无噪声地前向运行 duration 秒，求稳定平衡点 S*。
    若 max(abs(dS/dt)) <= tol，直接采用末态；否则用 root 数值精修。
    S 为初始状态，支持标量或 (N,)；多稳态时会影响分支选择。
    返回前检查残差、0<=S*<=1 及 Jacobian 特征值实部均为负。
    '''
    parameters, S, _, _ = check_model_parameters(struc_conn_matrix, G, w, I, tau_S, gamma, J_N, S=S)
    return _solve_S_star(parameters, S, duration, tol)


def _solve_S_star(parameters, S, duration=2.0, tol=1e-10):
    '''使用已统一形状的参数求解稳态，供 get_S_star 调用。'''
    if np.any(S < 0) or np.any(S > 1):
        raise ValueError('The initial state S must lie in [0,1].')
    if not np.isfinite([duration, tol]).all() or duration <= 0 or tol <= 0:
        raise ValueError('duration and tol must be finite and positive.')

    # 局部启用双精度，不改变调用者的 JAX/BrainPy 全局精度设置。
    with enable_x64():
        def drift(S):
            return DecoDrift(S, *parameters)

        def jacobian(S):
            return DecoDrift_jacobian(S, *parameters)

        warmup = solve_ivp(lambda t, S: drift(S), (0, duration), S, method='BDF',
                           jac=lambda t, S: jacobian(S), rtol=1e-8, atol=1e-10)
        if not warmup.success:
            raise RuntimeError('Equilibrium warm-up failed: ' + warmup.message)
        S = warmup.y[:, -1]
        message = warmup.message
        if np.max(np.abs(drift(S))) > tol:
            sol = root(drift, S, jac=jacobian, tol=1e-10)
            S, message = sol.x, sol.message
        if (not np.isfinite(S).all() or np.max(np.abs(drift(S))) > tol
                or np.any(S < 0) or np.any(S > 1)):
            raise RuntimeError('No physical equilibrium found; try another S. ' + message)
        Jacobian = np.asarray(jacobian(S))

    if np.max(np.linalg.eigvals(Jacobian).real) >= 0:
        raise ValueError('The equilibrium is unstable; try another S for a stable branch.')
    return S


def theory_ECFC(struc_conn_matrix, G, w, I, sigma, S_star,
                tau_S=0.1, gamma=0.641, J_N=0.2609,
                delta=1.0, return_info=False):
    '''
    在确定性平衡点附近线性化，计算神经状态 S 的理论 EC、FC。

    struc_conn_matrix : (N,N)，沿用 DecoModel 的 S @ struc_conn_matrix。
    G : 全局耦合，按原模型取 abs(G)。
    J_N : 突触耦合常数，默认 0.2609；与 Jacobian 矩阵区分。
    w, I, sigma : 标量、(1,) 或 (N,)；sigma 为独立白噪声幅度。
    S_star : 必须提前计算好的稳态，直接使用，不在本函数内求解或精修。
             检查有限值、物理范围和局部稳定性，残差随 return_info 返回。
    delta : 各源节点的状态脉冲幅度，默认单位脉冲；None 表示 1-S*。
            delta=None 仅为原仿真 S->1 重置的一阶近似。
    return_info : True 时额外返回 S、Jacobian、covariance、equilibrium_residual、S_source 字典。

    返回 EC, FC（NumPy 数组）；EC 行接收、列刺激，FC 为节点间相关矩阵。
    EC = integral_0^inf exp(Jacobian*t) dt @ diag(delta)
       = -Jacobian^(-1) @ diag(delta)，保留自响应。
    Jacobian*P + P*Jacobian.T + diag(sigma**2) = 0，FC 为 P 的相关系数。
    EC 是响应面积，未作原 def_bodyEC 的全时程平均或 rescale_EC。
    仅适用于稳定平衡点附近的小扰动，FC 为 S 的 Pearson FC。
    '''
    parameters, S_star, sigma, delta = check_model_parameters(
        struc_conn_matrix, G, w, I, tau_S, gamma, J_N,
        S=S_star, sigma=sigma, delta=delta)
    if np.any(S_star < 0) or np.any(S_star > 1):
        raise ValueError('S_star must lie in [0,1].')
    with enable_x64():
        Jacobian = np.asarray(DecoDrift_jacobian(S_star, *parameters))
        equilibrium_residual = float(np.max(np.abs(DecoDrift(S_star, *parameters))))
    if np.max(np.linalg.eigvals(Jacobian).real) >= 0:
        raise ValueError('The supplied S_star is locally unstable.')
    N = S_star.size

    # Green 函数从 0 积分到无穷；solve 代替显式求逆。
    EC = np.linalg.solve(Jacobian, -np.eye(N)) # EC = - mat_inverse(Jacobian)
    delta = 1-S_star if delta is None else delta
    EC = EC * delta[None, :]

    # 独立 Wiener 噪声：Q=diag(sigma**2)，此处不再乘 dt。
    Q = np.diag(sigma**2)
    P = solve_continuous_lyapunov(Jacobian, -Q)
    P = (P + P.T) / 2
    if np.any(np.diag(P) <= 0):
        raise ValueError('FC is undefined for nodes with zero stationary variance.')
    std = np.sqrt(np.diag(P))
    FC = P / np.outer(std, std)

    if return_info:
        return EC, FC, dict(S=S_star, Jacobian=Jacobian, covariance=P,
                           equilibrium_residual=equilibrium_residual,
                           S_source='provided')
    return EC, FC
