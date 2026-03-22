# ！/usr/bin/python
# -*- coding: utf-8 -*-#

'''
---------------------------------
 Name:         controllers.py
 Description:  Implement the solver-based agent of the proposed MASA framework.
 Author:       MASA
---------------------------------
'''

#from cvxopt import matrix, solvers
import cvxpy as cp
from scipy.linalg import sqrtm
import pandas as pd
import numpy as np
from scipy.integrate import quad
import scipy.stats as spstats
from scipy.stats import t
#solvers.options['show_progress'] = False
from joblib import Parallel, delayed
from sklearn.feature_selection import mutual_info_regression
from sklearn.covariance import ledoit_wolf, oas, ledoit_wolf_shrinkage
import os
import mlfinlab.ensemble


# def realized_covariance(date):
#     datas = pd.read_csv("/mnt/f/data/minute_300/300/{}.csv".format(date))
#     datas.set_index(['date','datetime','instrument'], inplace=True)
#     data = datas[['close']].apply(np.log)
#     pct_data = (
#         data.groupby('instrument')['close'].diff().dropna()
#     )
#     d1 = pct_data.unstack('datetime')
#     d2 = d1.T
#     rcov = d1.values@d2.values
#     np.savetxt(f'/mnt/f/data/rcov/{date}.txt', rcov, delimiter='\t')
#     return rcov
#
# def realized_covariance_1(datas,date):
#     # datas = pd.read_csv("/mnt/f/data/minute_300/FFD_300/{}.csv".format(date))
#     datas.set_index(['stock','datetime'], inplace=True)
#     # data = datas[['close']].apply(np.log)
#     # tmp = data.groupby('instrument')
#     # dfs = []
#     # for i, stock in tmp:
#     #     df = stock['close'].pct_change().dropna()
#     #     dfs.append(df)
#     # pct_data = pd.concat(dfs).sort_index()
#     # pct_data = (
#     #     data.groupby('instrument')['close'].diff().dropna()
#     # )
#     # d1 = pct_data.unstack('datetime')
#     # d2 = d1.T
#     # rcov = d1.values@d2.values
#     # q1=return_entry_1(d1.values)
#     d1 = datas['close'].unstack('datetime')
#     re_entry = return_entry(d1.values)
#     re_entry = pd.DataFrame(re_entry)
#     re_entry.columns = [date]
#     pct_data = datas.reset_index(level='stock').groupby('stock').first()
#     re_entry['stock'] = pct_data.index
#     re_entry.to_csv(f'/mnt/f/data/difference_entry_rolling/{date}.csv',index=False)
#     return re_entry
def pre_averaging_realized_covariance(date):
    c = 2
    datas = pd.read_csv("/mnt/f/data/minute_300/300/{}.csv".format(date))
    datas.set_index(['date', 'datetime', 'instrument'], inplace=True)
    data = datas[['close']].apply(np.log)
    g = lambda x: min(x, 1 - x)
    K_n = int(c * (240) ** 0.5)
    weights = np.array([g(j / K_n) for j in range(1, K_n + 1)])
    weights_matrix = weights[np.newaxis, :] * np.ones((300, 1))
    d1 = data.unstack('datetime')
    m, n = d1.shape
    # 按行计算
    dfs = []
    for j in range(n - K_n):
        diff = d1.values[:, j + 1: j + K_n + 1] - d1.values[:, j: j + K_n]  # 差分项
        Y = np.sum(weights_matrix * diff, axis=1)  # 按行加权求和
        dfs.append(Y)
    Y_hat = np.array(dfs)
    pcov = Y_hat.T @ Y_hat
    np.savetxt(f'/mnt/f/data/prcov/{date}.txt', pcov, delimiter='\t')

    return pcov


def improve_pre_averaging_realized_covariance(date):
    datas = pd.read_csv("/mnt/f/data/minute_300/300/{}.csv".format(date))
    c = 2
    datas.set_index(['date', 'datetime', 'instrument'], inplace=True)
    data = datas[['close']].apply(np.log)
    g = lambda x: min(x, 1 - x)
    g_squared = lambda t: g(t) ** 2
    phi, _ = quad(g_squared, 0, 1)
    K_n = int(c * (240) ** 0.5)
    weights = np.array([g(j / K_n) for j in range(1, K_n + 1)])
    # 计算相邻元素的差
    differences = np.diff(weights)
    # 计算差的平方和
    s = np.sum(differences ** 2)
    weights_matrix = weights[np.newaxis, :] * np.ones((300, 1))

    dfs = []
    d1 = data.unstack('datetime')
    m, n = d1.shape
    # 计算对数价格的差
    differences_p = np.diff(d1)
    # 计算对数价格差的平方和
    squared_p = np.sum(differences_p ** 2, axis=1) * (1 / (2 * n))
    n_hat = np.diag(squared_p)
    # 按行计算
    for j in range(n - K_n):
        diff = d1.values[:, j + 1: j + K_n + 1] - d1.values[:, j: j + K_n]  # 差分项
        Y = np.sum(weights_matrix * diff, axis=1)  # 按行加权求和
        dfs.append(Y)
    Y_hat = np.array(dfs)
    Ipcov = (1 / (phi * K_n)) * (Y_hat.T @ Y_hat - (n - K_n) * s * n_hat)
    np.savetxt(f'/mnt/f/data/iprcov/{date}.txt', Ipcov, delimiter='\t')

    return Ipcov


def RL_withoutController(a_rl, env=None):
    a_cbf = np.array([0] * env.config.topK)
    a_rl = np.array(a_rl)
    env.action_cbf_memeory.append(a_cbf)
    env.action_rl_memory.append(a_rl)
    a_final = a_rl + a_cbf
    return a_final

def RL_withController(a_rl, a_buffer=None,env=None, mask=None):
    a_rl = np.array(a_rl)
    env.action_rl_memory.append(a_rl)
    if env.config.pricePredModel == 'MA':
        # pred_prices_change = get_pred_price_change(env=env)
        # pred_dict = {'shortterm': pred_prices_change}
        pass
    else:
        raise ValueError("Cannot find the price prediction model [{}]..".format(env.config.pricePredModel))
    if env.config.gmv == 'cvar':
        if mask is None:
            a_cbf, is_solvable_status = gmv_cbf_opt(env=env, a_rl=a_rl, pred_dict=pred_dict)
        else:
            a_cbf, is_solvable_status = mask_cvar_cbf_opt(env=env, a_rl=a_rl, mask=mask)
    elif env.config.gmv == 'entry':
        a_cbf, is_solvable_status = mask_entry_opt(env=env, a_rl=a_rl, pred_dict=pred_dict, mask=mask)
    elif env.config.gmv == 'gmv':
        a_cbf, is_solvable_status = mask_gmv_cbf_opt(env=env, a_rl=a_rl, pred_dict=pred_dict, mask=mask)
    elif env.config.gmv == 'cov':
        if mask is None:
            a_cbf, is_solvable_status = cbf_opt(env=env, a_rl=a_rl, pred_dict=pred_dict)
        else:
            a_cbf, is_solvable_status = mask_cbf_opt(env=env, a_rl=a_rl, pred_dict=pred_dict, mask=mask)
    elif env.config.gmv == 'cvar-gmv':
        a_cbf, is_solvable_status = mask_cvar_gmv_cbf_opt(env=env, a_rl=a_rl, pred_dict=pred_dict, mask=mask)

    cur_dcm_weight = 1.0
    cur_rl_weight = 1.0
    if is_solvable_status:
        a_cbf_weighted = a_cbf * cur_dcm_weight
        env.action_cbf_memeory.append(a_cbf_weighted)
        a_rl_weighted = a_rl * cur_rl_weight
        a_final = a_rl_weighted + a_cbf_weighted
    else:
        env.action_cbf_memeory.append(
            np.array([0] * (env.config.topK + 1)) if a_rl.shape[0] != env.config.topK else np.array(
                [0] * env.config.topK))
        a_final = a_rl
    return a_final


def get_pred_price_change(env):
    ma_lst = env.ctl_state['MA-{}'.format(env.config.otherRef_indicator_ma_window)]
    pred_prices = ma_lst
    cur_close_price = np.array(env.curData['close'].values)
    pred_prices_change = (pred_prices - cur_close_price) / cur_close_price
    return pred_prices_change


# def entry_risk_obj(V,w):
#     eigenvalues, eigenvectors = np.linalg.eigh(V)
#     A = np.maximum(eigenvalues.reshape(-1, 1), 1e-6)
#     W = eigenvectors  # 特征向量矩阵 W
#     f = W.T @ w
#     f_2 = cp.square(f)
#     o = cp.multiply(A,f_2)
#     obj1 = o/cp.sum(o)
#     H = 1-(1/A.shape[0])*cp.exp(-cp.sum(cp.multiply(obj1,cp.log(obj1))))
#     return H
#
def process_row(row):
    try:
        info = mf.sigma_mapping(row, np.std(row))
        message = mf.encode_array(row, info)
        entry = mf.get_konto_entropy(message)
        return entry
    except Exception as e:
        return 0
#
# def return_entry(daily_return):
#     # daily_return 应为一个二维数组，行对应不同资产
#     n_assets = daily_return.shape[0]
#     # 使用所有可用的CPU核心进行并行处理
#     results = Parallel(n_jobs=-1)(delayed(process_row)(daily_return[i, :]) for i in range(n_assets))
#     # 返回结果为一维数组，每个元素对应一个资产的熵
#     return np.array(results)
#
# # 计算 MI（仅计算上三角部分）
# def compute_mi(returns, i, j):
#     return mutual_info_regression(returns[i, :].reshape(-1, 1), returns[j, :])[0]
#
# def compute_self_mi(returns, i):
#     return mutual_info_regression(returns[i, :].reshape(-1, 1), returns[i, :])[0]
#
# def return_entry_info(date):
#
#     datas = pd.read_csv("/mnt/f/data/minute_300/FFD_300/{}.csv".format(date))
#     datas.set_index(['date','datetime','instrument'], inplace=True)
#     data = datas[['close']].apply(np.log)
#     pct_data = (
#         data.groupby('instrument')['close'].diff().dropna()
#     )
#     d1 = pct_data.unstack('datetime')
#     n_assets = len(d1)
#     mi_matrix = np.zeros((n_assets, n_assets))
#     # 并行计算对角线（自身 MI）
#     self_mi_values = Parallel(n_jobs=-1)(delayed(compute_self_mi)(d1.values, i) for i in range(n_assets))
#     # 并行计算上三角部分（两两 MI）
#     triu_indices = np.triu_indices(n_assets, k=1)  # 获取上三角索引
#     # 并行计算 MI
#     mi_values = Parallel(n_jobs=-1)(delayed(compute_mi)(d1.values, i, j) for i, j in zip(*triu_indices))
#     # 填充上三角
#     mi_matrix[triu_indices] = mi_values
#
#     # 复制到下三角
#     mi_matrix += mi_matrix.T
#     # # 填充对角线
#     np.fill_diagonal(mi_matrix, self_mi_values)
#
#     np.savetxt(f'/mnt/f/data/mi_difference_entry_cov/{date}.txt', mi_matrix, delimiter='\t')
#
#     return mi_matrix
#
# def return_entry_info_1(date):
#
#     datas = pd.read_csv("/mnt/f/data/minute_30/{}.csv".format(date))
#     datas.set_index(['date','datetime','instrument'], inplace=True)
#     data = datas[['close']].apply(np.log)
#     pct_data = (
#         data.groupby('instrument')['close'].diff().dropna()
#     )
#     d1 = pct_data.unstack('datetime')
#     n_assets = len(d1)
#     mi_matrix = np.zeros((n_assets, n_assets))
#     # 并行计算对角线（自身 MI）
#     self_mi_values = Parallel(n_jobs=-1)(delayed(compute_self_mi)(d1.values, i) for i in range(n_assets))
#     # 并行计算上三角部分（两两 MI）
#     triu_indices = np.triu_indices(n_assets, k=1)  # 获取上三角索引
#     # 并行计算 MI
#     mi_values = Parallel(n_jobs=-1)(delayed(compute_mi)(d1.values, i, j) for i, j in zip(*triu_indices))
#     # 填充上三角
#     mi_matrix[triu_indices] = mi_values
#
#     # 复制到下三角
#     mi_matrix += mi_matrix.T
#     # # 填充对角线
#     np.fill_diagonal(mi_matrix, self_mi_values)
#
#     np.savetxt(f'/mnt/f/data/mi_crp_entry_cov/{date}.txt', mi_matrix, delimiter='\t')
#
#     return mi_matrix
# def return_entry_1(daily_return):
#     # daily_return 应为一个二维数组，行对应不同资产
#     n_assets = daily_return.shape[0]
#     results = []
#     # 使用所有可用的CPU核心进行并行处理
#     for i in range(n_assets):
#         re = process_row(daily_return[i, :])
#         results.append(re)
#     # 返回结果为一维数组，每个元素对应一个资产的熵
#     return np.array(results)

def cbf_opt(env, a_rl, pred_dict):
    """
    The risk constraint is based on controller barrier function (CBF) method. Not just considering the satisfaction of the current risk constraint, but also considering the trends/gradients of the future risk.
    """
    pred_prices_change = pred_dict['shortterm']

    a_rl = np.array(a_rl)
    assert np.sum(
        np.abs(a_rl)) - 1 < 0.0001, "The sum of a_rl is not equal to 1, the value is {}, a_rl list: {}".format(
        np.sum(a_rl), a_rl)
    N = env.config.topK
    # Past N days daily return rate of each stock, (num_of_stocks, lookback_days), [[t-N+1, t-N+2, .., t-1, t]]
    daily_return_ay = env.ctl_state['DAILYRETURNS-{}'.format(env.config.dailyRetun_lookback)]
    daily_return_ay = np.array([np.fromstring(row.strip("[]"), sep=" ") for row in daily_return_ay])
    # re_entry = entry(daily_return_ay[:,-1])
    cov_r_t0 = np.cov(daily_return_ay)
    w_t0 = np.array([env.actions_memory[-1]])[:, 1:] if a_rl.shape[0] != N else np.array([env.actions_memory[-1]])
    try:
        risk_stg_t0 = np.sqrt(np.matmul(np.matmul(w_t0, cov_r_t0), w_t0.T)[0][0])
    except Exception as error:
        print("Risk-(MV model variance): {}".format(error))
        risk_stg_t0 = 0
    risk_market_t0 = env.config.risk_market
    if len(env.risk_adj_lst) <= 1:
        risk_safe_t0 = env.risk_adj_lst[-1]
    else:
        if env.is_last_ctrl_solvable:
            risk_safe_t0 = env.risk_adj_lst[-2]
        else:
            risk_safe_t0 = risk_stg_t0 + risk_market_t0

    gamma = env.config.cbf_gamma
    risk_market_t1 = env.config.risk_market
    risk_safe_t1 = env.risk_adj_lst[-1]

    pred_prices_change_reshape = np.reshape(pred_prices_change, (-1, 1))
    r_t1 = np.append(daily_return_ay[:, 1:], pred_prices_change_reshape, axis=1)

    cov_r_t1 = np.cov(r_t1)
    cov_sqrt_t1 = sqrtm(cov_r_t1)
    cov_sqrt_t1 = cov_sqrt_t1.real
    G_ay = np.array([]).reshape(-1, N)

    h_0 = np.array([])

    use_cvxopt_threshold = 10  # using cvxopt tool will be faster when the size of portfolio is less than or equal to 10. Otherwise, the cvxpy tool will be faster.
    w_lb = 0
    w_ub = 1

    if env.config.topK <= use_cvxopt_threshold:
        # Implemented by cvxopt
        A_eq = np.array([]).reshape(-1, N)
        linear_g1 = np.array([[1.0] * N])  # (1, N)
        A_eq = np.append(A_eq, linear_g1, axis=0)
        A_eq = matrix(A_eq)
        b_eq = np.array([0.0])
        b_eq = matrix(b_eq)

        h_0 = np.append(h_0, a_rl, axis=0)  # linear_h3, 0 <= (a_RL + a_cbf)
        h_0 = np.append(h_0, 1 - a_rl, axis=0)  # linear_h4 (a_RL + a_cbf) <= 1

        linear_g3 = np.diag([-1.0] * N)
        G_ay = np.append(G_ay, linear_g3, axis=0)  # 0 <= (a_RL + a_cbf)
        linear_g4 = np.diag([1.0] * N)
        G_ay = np.append(G_ay, linear_g4, axis=0)  # (a_RL + a_cbf) <= 1

    else:
        a_rl_re_sign = np.reshape(a_rl, (-1, 1))
        sign_mul = np.ones((1, N + 1 if a_rl.shape[0] != N else N))
        w_lb_sign = w_lb
        w_ub_sign = w_ub

    last_h_risk = (-risk_market_t0 - risk_stg_t0 + risk_safe_t0)
    last_h_risk = np.max([last_h_risk, 0.0])
    socp_d = -risk_market_t1 + risk_safe_t1 + (gamma - 1) * last_h_risk  # 指定最小风险

    step_add_lst = [0.002, 0.002, 0.002, 0.002, 0.002, 0.005, 0.005, 0.005, 0.005, 0.005]
    cnt = 1
    if env.config.is_enable_dynamic_risk_bound:
        cnt_th = env.config.ars_trial  # Iterative risk relaxation
    else:
        cnt_th = 1

    if env.config.topK <= use_cvxopt_threshold:
        # Implemented by cvxopt
        socp_b = np.matmul(cov_sqrt_t1, a_rl)
        h = np.append(h_0, [socp_d], axis=0)  # socp_d
        h = np.append(h, socp_b, axis=0)  # socp_b
        h = matrix(h)
        socp_cx = np.array([[0.0] * N])
        G_ay = np.append(G_ay, -socp_cx, axis=0)
        G_ay = np.append(G_ay, -cov_sqrt_t1, axis=0)  # socp_ax
        G = matrix(G_ay)  # G = matrix(np.transpose(np.transpose(G_ay)))

        linear_eq_num = 2 * N
        dims = {'l': linear_eq_num, 'q': [N + 1], 's': []}
        QP_P = matrix(np.eye(N)) * 2  # (1/2) xP'x
        QP_Q = matrix(np.zeros((N, 1)))  # q'x
        while cnt <= cnt_th:
            try:
                sol = solvers.coneqp(QP_P, QP_Q, G, h, dims, A_eq, b_eq)
                if sol['status'] == 'optimal':
                    solver_flag = True
                    break
                else:
                    raise
            except:
                solver_flag = False
                cnt += 1
                risk_safe_t1 = risk_safe_t1 + step_add_lst[cnt - 2]
                socp_d = -risk_market_t1 + risk_safe_t1 + (gamma - 1) * (-risk_market_t0 - risk_stg_t0 + risk_safe_t0)
                h = np.append(h_0, [socp_d], axis=0)  # socp_d
                h = np.append(h, socp_b, axis=0)  # socp_b
                h = matrix(h)

        if solver_flag:
            if sol['status'] == 'optimal':
                a_cbf = np.reshape(np.array(sol['x']), -1)
                env.solver_stat['solvable'] = env.solver_stat['solvable'] + 1
                is_solvable_status = True
                env.risk_adj_lst[-1] = risk_safe_t1
                # Check the solution whether satisfy the risk constraint.
                cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl + a_cbf), cov_r_t1), (a_rl + a_cbf).T))
                assert (cur_alpha_risk - socp_d) <= 0.00001, 'cur risk: {}, socp_d {}'.format(cur_alpha_risk, socp_d)
                assert np.abs(np.sum(np.abs((a_rl + a_cbf))) - 1) <= 0.00001, 'sum of actions: {} \n{} \n{}'.format(
                    np.sum(np.abs((a_rl + a_cbf))), a_rl, a_cbf)
                env.solvable_flag.append(0)
            else:
                a_cbf = np.zeros(N)
                env.solver_stat['insolvable'] = env.solver_stat['insolvable'] + 1
                is_solvable_status = False
                cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl), cov_r_t1), (a_rl).T))
                env.solvable_flag.append(1)
        else:
            a_cbf = np.zeros(N)
            env.solver_stat['insolvable'] = env.solver_stat['insolvable'] + 1
            is_solvable_status = False
            cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl), cov_r_t1), (a_rl).T))
            env.solvable_flag.append(1)
            # print("Failed to solve the problem.")

    else:
        # Complete solver
        # ++ Implemented by cvxpy
        if a_rl.shape[0] != N:
            cp_x = cp.Variable((N + 1, 1))
            a_rl_re = np.reshape(a_rl, (-1, 1))
            cp_constraint = []
            cp_constraint.append(cp.sum(sign_mul @ cp_x) + cp.sum(a_rl_re_sign) == 1)
            cp_constraint.append(a_rl_re + cp_x >= w_lb_sign)
            cp_constraint.append(a_rl_re + cp_x <= w_ub_sign)
            cp_constraint.append(cp.SOC(socp_d, cov_sqrt_t1 @ (a_rl_re[1:] + cp_x[1:])))
        else:
            cp_x = cp.Variable((N, 1))
            a_rl_re = np.reshape(a_rl, (-1, 1))
            cp_constraint = []
            cp_constraint.append(cp.sum(sign_mul @ cp_x) + cp.sum(a_rl_re_sign) == 1)
            cp_constraint.append(a_rl_re + cp_x >= w_lb_sign)
            cp_constraint.append(a_rl_re + cp_x <= w_ub_sign)
            cp_constraint.append(cp.SOC(socp_d, cov_sqrt_t1 @ (a_rl_re + cp_x)))
        # cvxpy
        while cnt <= cnt_th:
            try:
                obj_f2 = cp.sum_squares(cp_x)
                cp_obj = cp.Minimize(obj_f2)
                cp_prob = cp.Problem(cp_obj, cp_constraint)
                cp_prob.solve(solver=cp.ECOS, verbose=False)

                if cp_prob.status == 'optimal':
                    solver_flag = True
                    break
                else:
                    raise
            except:
                solver_flag = False
                cnt += 1
                risk_safe_t1 = risk_safe_t1 + step_add_lst[cnt - 2]
                socp_d = -risk_market_t1 + risk_safe_t1 + (gamma - 1) * last_h_risk
                if a_rl.shape[0] != N:
                    cp_constraint[-1] = cp.SOC(socp_d, cov_sqrt_t1 @ (a_rl_re[1:] + cp_x[1:]))
                else:
                    cp_constraint[-1] = cp.SOC(socp_d, cov_sqrt_t1 @ (a_rl_re + cp_x))

        if (cp_prob.status == 'optimal') and solver_flag:
            is_solvable_status = True
            a_cbf = np.reshape(np.array(cp_x.value), -1)
            env.solver_stat['solvable'] = env.solver_stat['solvable'] + 1
            # Check the solution whether satisfy the risk constraint.
            env.risk_adj_lst[-1] = risk_safe_t1
            # 策略风险
            if a_rl.shape[0] != N:
                cur_alpha_risk = np.sqrt(
                    np.matmul(np.matmul((a_rl[1:] + a_cbf[1:]), cov_r_t1), (a_rl[1:] + a_cbf[1:]).T))
            else:
                cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl + a_cbf), cov_r_t1), (a_rl + a_cbf).T))
            assert (cur_alpha_risk - socp_d) <= 0.00001, 'cur risk: {}, socp_d {}'.format(cur_alpha_risk, socp_d)
            assert np.abs(np.sum(np.abs(a_rl + a_cbf)) - 1) <= 0.00001, 'sum of actions: {} \n{} \n{}'.format(
                np.sum(np.abs((a_rl + a_cbf))), a_rl, a_cbf)
            env.solvable_flag.append(0)
        else:
            if a_rl.shape[0] != N:
                a_cbf = np.zeros(N + 1)
                cur_alpha_risk = np.sqrt(
                    np.matmul(np.matmul((a_rl[1:] + a_cbf[1:]), cov_r_t1), (a_rl[1:] + a_cbf[1:]).T))
            else:
                a_cbf = np.zeros(N)
                cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl), cov_r_t1), (a_rl).T))

            env.solver_stat['insolvable'] = env.solver_stat['insolvable'] + 1
            is_solvable_status = False
            env.solvable_flag.append(1)
            env.risk_adj_lst[-1] = risk_safe_t1
    env.risk_pred_lst.append(cur_alpha_risk)
    env.is_last_ctrl_solvable = is_solvable_status
    if not is_solvable_status:
        print("无解")
    if cnt > 1:
        env.stepcount = env.stepcount + 1
    # 求解器解出的action
    return a_cbf, is_solvable_status


def mask_cbf_opt(env, a_rl, pred_dict, mask):
    """
    The risk constraint is based on controller barrier function (CBF) method. Not just considering the satisfaction of the current risk constraint, but also considering the trends/gradients of the future risk.

    """
    pred_prices_change = pred_dict['shortterm']
    # cur_date_1 = str(env.curData['date'].unique()[0]).split('T')[0]
    a_rl = np.array(a_rl)
    assert np.sum(
        np.abs(a_rl)) - 1 < 0.0001, "The sum of a_rl is not equal to 1, the value is {}, a_rl list: {}".format(
        np.sum(a_rl), a_rl)
    N = env.config.topK
    # Past N days daily return rate of each stock, (num_of_stocks, lookback_days), [[t-N+1, t-N+2, .., t-1, t]]
    daily_return_ay = env.ctl_state['DAILYRETURNS-{}'.format(env.config.dailyRetun_lookback)]
    daily_return_ay = np.array([np.fromstring(row.strip("[]"), sep=" ") for row in daily_return_ay])

    cov_r_t0 = np.cov(daily_return_ay)
    # cov_r_t0, shrinkage_coef = ledoit_wolf(cov_r_t0)
    if a_rl.shape[0] != N:
        mask = np.hstack(([[0]], mask))
        w_t0 = (np.array([env.actions_memory[-1]]) * (1 - mask))[:, 1:]
    else:
        mask = np.array(mask)
        w_t0 = np.array([env.actions_memory[-1]]) * (1 - mask)

    try:
        risk_stg_t0 = np.sqrt(np.matmul(np.matmul(w_t0, cov_r_t0), w_t0.T)[0][0])
    except Exception as error:
        print("Risk-(MV model variance): {}".format(error))
        risk_stg_t0 = 0
    risk_market_t0 = env.config.risk_market
    if len(env.risk_adj_lst) <= 1:
        risk_safe_t0 = env.risk_adj_lst[-1]
    else:
        if env.is_last_ctrl_solvable:
            risk_safe_t0 = env.risk_adj_lst[-2]
        else:
            risk_safe_t0 = risk_stg_t0 + risk_market_t0

    gamma = env.config.cbf_gamma
    risk_market_t1 = env.config.risk_market
    risk_safe_t1 = env.risk_adj_lst[-1]

    pred_prices_change_reshape = np.reshape(pred_prices_change, (-1, 1))
    r_t1 = np.append(daily_return_ay[:, 1:], pred_prices_change_reshape, axis=1)
    # pred_entry=return_entry(r_t1)
    cov_r_t1 = np.cov(r_t1)
    # cov_r_t1, shrinkage_coef = ledoit_wolf(cov_r_t1)
    cov_sqrt_t1 = sqrtm(cov_r_t1)
    cov_sqrt_t1 = cov_sqrt_t1.real
    cur_date = env.curData['date'].unique()[0]
    # return_entry_t1 = pd.read_csv(f"/mnt/f/data/difference_entry_rolling/{cur_date}.csv")['rolling_entry_5'].values.T
    # price_entry_t1 = pd.read_csv(f"/mnt/f/data/difference_entry_rolling/{cur_date}.csv")['rolling_difference_entry_5'].values.T

    # use_cvxopt_threshold = 10  # using cvxopt tool will be faster when the size of portfolio is less than or equal to 10. Otherwise, the cvxpy tool will be faster.
    w_lb = 0
    w_ub = 1

    a_rl_re_sign = np.reshape(a_rl, (-1, 1))
    sign_mul = np.ones((1, N + 1 if a_rl.shape[0] != N else N))
    w_lb_sign = w_lb
    w_ub_sign = w_ub

    last_h_risk = (-risk_market_t0 - risk_stg_t0 + risk_safe_t0)
    last_h_risk = np.max([last_h_risk, 0.0])
    socp_d = -risk_market_t1 + risk_safe_t1 + (gamma - 1) * last_h_risk  # 指定最小风险

    step_add_lst = [0.002, 0.002, 0.002, 0.002, 0.002, 0.005, 0.005, 0.01, 0.03, 0.05]
    cnt = 1
    if env.config.is_enable_dynamic_risk_bound:
        cnt_th = env.config.ars_trial  # Iterative risk relaxation
    else:
        cnt_th = 1

    if True:
        # Complete solver
        # ++ Implemented by cvxpy
        if a_rl.shape[0] != N:
            cp_x = cp.Variable((N + 1, 1))
            a_rl_re = np.reshape(a_rl, (-1, 1))
            cp_constraint = []
            num_count = np.count_nonzero(mask == 0)
            weight_safe = 1 / num_count
            mask = (1 - mask)
            cp_constraint.append(a_rl_re[0] + cp_x[0] <= weight_safe)
            # cp_constraint.append(cp_x >= -0.1)
            cp_constraint.append(cp.sum((sign_mul * mask) @ cp_x) + cp.sum(a_rl_re_sign) == 1)
            cp_constraint.append(a_rl_re + cp.multiply(mask.T, cp_x) >= w_lb_sign)
            cp_constraint.append(a_rl_re + cp.multiply(mask.T, cp_x) <= w_ub_sign)
            cp_constraint.append(cp_x == cp.multiply(mask.T, cp_x))
            cp_constraint.append(cp.SOC(socp_d, cov_sqrt_t1 @ (a_rl_re[1:] + cp.multiply(mask.T, cp_x)[1:])))
            # cp_constraint.append(return_entry_t1 @ (a_rl_re[1:] + cp.multiply(mask.T,cp_x)[1:])<=1.3)
            # obj_f1 = entry(cov_r_t1, cp_x[1:])
            # obj_f1 = pred_entry@cp.multiply(mask.T,cp_x)[1:]
        else:
            cp_x = cp.Variable((N, 1))
            a_rl_re = np.reshape(a_rl, (-1, 1))
            cp_constraint = []
            num_count = np.count_nonzero(mask == 0)
            weight_safe = 1 / num_count
            mask = (1 - mask)
            cp_constraint.append(a_rl_re + cp_x <= 0.1)
            # cp_constraint.append(cp_x >= -0.1)
            cp_constraint.append(cp.sum((sign_mul * mask) @ cp_x) + cp.sum(a_rl_re_sign) == 1)
            cp_constraint.append(a_rl_re + cp.multiply(mask.T, cp_x) >= w_lb_sign)
            cp_constraint.append(a_rl_re + cp.multiply(mask.T, cp_x) <= w_ub_sign)
            cp_constraint.append(cp_x == cp.multiply(mask.T, cp_x))
            cp_constraint.append(cp.SOC(socp_d, cov_sqrt_t1 @ (a_rl_re + cp.multiply(mask.T, cp_x))))
            # cp_constraint.append(return_entry_t1 @ (a_rl_re + cp.multiply(mask.T,cp_x))<=1.3)

            # obj_f1 = entry(cov_r_t1, cp_x)
            # obj_f1 = pred_entry@cp_x
        # cvxpy
        while cnt <= cnt_th:
            try:
                # price_entropy = cp_x[1:].T @ price_entry_t1
                obj_f2 = cp.sum_squares(cp_x)
                cp_obj = cp.Minimize(obj_f2)
                cp_prob = cp.Problem(cp_obj, cp_constraint)
                cp_prob.solve(solver=cp.ECOS, verbose=False)

                if cp_prob.status == 'optimal':
                    solver_flag = True
                    break
                else:
                    raise
            except Exception as e:
                # print(e)
                solver_flag = False
                cnt += 1
                risk_safe_t1 = risk_safe_t1 + step_add_lst[cnt - 2]
                socp_d = -risk_market_t1 + risk_safe_t1 + (gamma - 1) * last_h_risk
                if a_rl.shape[0] != N:
                    cp_constraint[-1] = cp.SOC(socp_d, cov_sqrt_t1 @ (a_rl_re[1:] + cp.multiply(mask.T, cp_x)[1:]))
                else:
                    cp_constraint[-1] = cp.SOC(socp_d, cov_sqrt_t1 @ (a_rl_re + cp.multiply(mask.T, cp_x)))

        if (cp_prob.status == 'optimal') and solver_flag:
            is_solvable_status = True
            a_cbf = np.reshape(np.array(cp_x.value), -1)
            env.solver_stat['solvable'] = env.solver_stat['solvable'] + 1
            # Check the solution whether satisfy the risk constraint.
            env.risk_adj_lst[-1] = risk_safe_t1
            # 策略风险
            if a_rl.shape[0] != N:
                cur_alpha_risk = np.sqrt(
                    np.matmul(np.matmul((a_rl[1:] + a_cbf[1:]), cov_r_t1), (a_rl[1:] + a_cbf[1:]).T))
            else:
                cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl + a_cbf), cov_r_t1), (a_rl + a_cbf).T))
            assert (cur_alpha_risk - socp_d) <= 0.00001, 'cur risk: {}, socp_d {}'.format(cur_alpha_risk, socp_d)
            assert np.abs(np.sum(np.abs(a_rl + a_cbf)) - 1) <= 0.00001, 'sum of actions: {} \n{} \n{}'.format(
                np.sum(np.abs((a_rl + a_cbf))), a_rl, a_cbf)
            env.solvable_flag.append(0)
        else:
            if a_rl.shape[0] != N:
                a_cbf = np.zeros(N + 1)
                cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl[1:]), cov_r_t1), (a_rl[1:]).T))
            else:
                a_cbf = np.zeros(N)
                cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl), cov_r_t1), (a_rl).T))

            env.solver_stat['insolvable'] = env.solver_stat['insolvable'] + 1
            is_solvable_status = False
            env.solvable_flag.append(1)
            env.risk_adj_lst[-1] = risk_safe_t1
    env.risk_pred_lst.append(cur_alpha_risk)
    env.is_last_ctrl_solvable = is_solvable_status
    if not is_solvable_status:
        print(cur_date, "无解")
    if cnt > 1:
        env.stepcount = env.stepcount + 1
    # 求解器解出的action
    return a_cbf, is_solvable_status


def gmv_cbf_opt(env, a_rl, pred_dict):
    """
    The risk constraint is based on GMV-CVaR method. Not just considering the satisfaction of the current risk constraint, but also considering the trends/gradients of the future risk.
    """
    # pred_prices_change = pred_dict['shortterm']
    # cur_date_1 = str(env.curData['date'].unique()[0]).split('T')[0]
    a_rl = np.array(a_rl)
    assert np.sum(
        np.abs(a_rl)) - 1 < 0.0001, "The sum of a_rl is not equal to 1, the value is {}, a_rl list: {}".format(
        np.sum(a_rl), a_rl)
    N = env.config.topK
    # Past N days daily return rate of each stock, (num_of_stocks, lookback_days), [[t-N+1, t-N+2, .., t-1, t]]
    daily_return_ay = env.ctl_state['DAILYRETURNS-{}'.format(env.config.dailyRetun_lookback)]
    daily_return_ay = np.array([np.fromstring(row.strip("[]"), sep=" ") for row in daily_return_ay])[-1]
    day = 1
    while True:
        try:
            cur_date = env.curData['date'].unique()[0]
            cur_date_0 = str(np.datetime64(cur_date) - np.timedelta64(day, 'D')).split('T')[0]
            # cov_r_t0 = realized_covariance(cur_date_0)
            cov_r_t0 = np.loadtxt(f"/mnt/f/data/rcov/{cur_date_0}.txt")
            break
        except:
            # print("当前日期无数据,调整到上一日")
            day += 1
            cur_date_0 = str(np.datetime64(cur_date) - np.timedelta64(day, 'D')).split('T')[0]

    # 前一步权重
    w_t0 = np.array([env.actions_memory[-1]])[:, 1:] if a_rl.shape[0] != N else np.array([env.actions_memory[-1]])
    try:
        risk_stg_t0 = np.sqrt(np.matmul(np.matmul(w_t0, cov_r_t0), w_t0.T)[0][0])
    except Exception as error:
        print("Risk-(MV model variance): {}".format(error))
        risk_stg_t0 = 0
    risk_market_t0 = env.config.risk_market
    if len(env.risk_adj_lst) <= 1:
        risk_safe_t0 = env.risk_adj_lst[-1]
    else:
        if env.is_last_ctrl_solvable:
            risk_safe_t0 = env.risk_adj_lst[-2]
        else:
            risk_safe_t0 = risk_stg_t0 + risk_market_t0

    gamma = env.config.cbf_gamma
    risk_market_t1 = env.config.risk_market
    risk_safe_t1 = env.risk_adj_lst[-1]

    # pred_prices_change_reshape = np.reshape(pred_prices_change, (-1, 1))
    # r_t1 = np.append(daily_return_ay[:, 1:], pred_prices_change_reshape, axis=1)
    #
    # cov_r_t1 = np.cov(r_t1)
    # cov_r_t1 = realized_covariance(cur_date)
    cov_r_t1 = np.loadtxt(f"/mnt/f/data/rcov/{cur_date}.txt")
    cov_sqrt_t1 = sqrtm(cov_r_t1)
    cov_sqrt_t1 = cov_sqrt_t1.real
    G_ay = np.array([]).reshape(-1, N)
    # if not np.all(np.linalg.eigvals(cov_r_t1) > 0):
    #     print("cov_r_t1 is not positive definite!")
    h_0 = np.array([])

    use_cvxopt_threshold = 10  # using cvxopt tool will be faster when the size of portfolio is less than or equal to 10. Otherwise, the cvxpy tool will be faster.
    w_lb = 0
    w_ub = 1

    if env.config.topK <= use_cvxopt_threshold:
        # Implemented by cvxopt
        A_eq = np.array([]).reshape(-1, N)
        linear_g1 = np.array([[1.0] * N])  # (1, N)
        A_eq = np.append(A_eq, linear_g1, axis=0)
        A_eq = matrix(A_eq)
        b_eq = np.array([0.0])
        b_eq = matrix(b_eq)

        h_0 = np.append(h_0, a_rl, axis=0)  # linear_h3, 0 <= (a_RL + a_cbf)
        h_0 = np.append(h_0, 1 - a_rl, axis=0)  # linear_h4 (a_RL + a_cbf) <= 1

        linear_g3 = np.diag([-1.0] * N)
        G_ay = np.append(G_ay, linear_g3, axis=0)  # 0 <= (a_RL + a_cbf)
        linear_g4 = np.diag([1.0] * N)
        G_ay = np.append(G_ay, linear_g4, axis=0)  # (a_RL + a_cbf) <= 1

    else:
        a_rl_re_sign = np.reshape(a_rl, (-1, 1))
        sign_mul = np.ones((1, N + 1 if a_rl.shape[0] != N else N))
        w_lb_sign = w_lb
        w_ub_sign = w_ub

    last_h_risk = (-risk_market_t0 - risk_stg_t0 + risk_safe_t0)
    last_h_risk = np.max([last_h_risk, 0.0])
    socp_d = -risk_market_t1 + risk_safe_t1 + (gamma - 1) * last_h_risk  # 指定最小风险

    step_add_lst = [0.002, 0.002, 0.002, 0.002, 0.002, 0.005, 0.005, 0.005, 0.005, 0.005]
    cnt = 1
    if env.config.is_enable_dynamic_risk_bound:
        cnt_th = env.config.ars_trial  # Iterative risk relaxation
    else:
        cnt_th = 1

    if env.config.topK <= use_cvxopt_threshold:
        # Implemented by cvxopt
        socp_b = np.matmul(cov_sqrt_t1, a_rl)
        h = np.append(h_0, [socp_d], axis=0)  # socp_d
        h = np.append(h, socp_b, axis=0)  # socp_b
        h = matrix(h)
        socp_cx = np.array([[0.0] * N])
        G_ay = np.append(G_ay, -socp_cx, axis=0)
        G_ay = np.append(G_ay, -cov_sqrt_t1, axis=0)  # socp_ax
        G = matrix(G_ay)  # G = matrix(np.transpose(np.transpose(G_ay)))

        linear_eq_num = 2 * N
        dims = {'l': linear_eq_num, 'q': [N + 1], 's': []}
        QP_P = matrix(np.eye(N)) * 2  # (1/2) xP'x
        QP_Q = matrix(np.zeros((N, 1)))  # q'x
        while cnt <= cnt_th:
            try:
                sol = solvers.coneqp(QP_P, QP_Q, G, h, dims, A_eq, b_eq)
                if sol['status'] == 'optimal':
                    solver_flag = True
                    break
                else:
                    raise
            except:
                solver_flag = False
                cnt += 1
                risk_safe_t1 = risk_safe_t1 + step_add_lst[cnt - 2]
                socp_d = -risk_market_t1 + risk_safe_t1 + (gamma - 1) * (-risk_market_t0 - risk_stg_t0 + risk_safe_t0)
                h = np.append(h_0, [socp_d], axis=0)  # socp_d
                h = np.append(h, socp_b, axis=0)  # socp_b
                h = matrix(h)

        if solver_flag:
            if sol['status'] == 'optimal':
                a_cbf = np.reshape(np.array(sol['x']), -1)
                env.solver_stat['solvable'] = env.solver_stat['solvable'] + 1
                is_solvable_status = True
                env.risk_adj_lst[-1] = risk_safe_t1
                # Check the solution whether satisfy the risk constraint.
                cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl + a_cbf), cov_r_t1), (a_rl + a_cbf).T))
                assert (cur_alpha_risk - socp_d) <= 0.00001, 'cur risk: {}, socp_d {}'.format(cur_alpha_risk, socp_d)
                assert np.abs(np.sum(np.abs((a_rl + a_cbf))) - 1) <= 0.00001, 'sum of actions: {} \n{} \n{}'.format(
                    np.sum(np.abs((a_rl + a_cbf))), a_rl, a_cbf)
                env.solvable_flag.append(0)
            else:
                a_cbf = np.zeros(N)
                env.solver_stat['insolvable'] = env.solver_stat['insolvable'] + 1
                is_solvable_status = False
                cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl), cov_r_t1), (a_rl).T))
                env.solvable_flag.append(1)
        else:
            a_cbf = np.zeros(N)
            env.solver_stat['insolvable'] = env.solver_stat['insolvable'] + 1
            is_solvable_status = False
            cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl), cov_r_t1), (a_rl).T))
            env.solvable_flag.append(1)
            # print("Failed to solve the problem.")
    else:
        # Complete solver
        # ++ Implemented by cvxpy
        var = env.var_lst[-1]
        if a_rl.shape[0] != N:
            cp_x = cp.Variable((N + 1, 1))
            z_i = cp.Variable((1, 1), nonneg=True)  # J 维辅助变量 z
            a_rl_re = np.reshape(a_rl, (-1, 1))
            cp_constraint = []
            cp_constraint.append(cp.sum(sign_mul @ cp_x) + cp.sum(a_rl_re_sign) == 1)
            cp_constraint.append(a_rl_re + cp_x >= w_lb_sign)
            cp_constraint.append(a_rl_re + cp_x <= w_ub_sign)
            tmp = [var + (a_rl_re[1:] + cp_x[1:]).T @ daily_return_ay + z_i >= 0]
            cp_constraint += tmp
            cp_constraint.append(cp.SOC(socp_d, cov_sqrt_t1 @ (a_rl_re[1:] + cp_x[1:])))
        else:
            cp_x = cp.Variable((N, 1))
            z_i = cp.Variable((1, 1), nonneg=True)  # J 维辅助变量 z
            a_rl_re = np.reshape(a_rl, (-1, 1))
            cp_constraint = []
            cp_constraint.append(cp.sum(sign_mul @ cp_x) + cp.sum(a_rl_re_sign) == 1)
            cp_constraint.append(a_rl_re + cp_x >= w_lb_sign)
            cp_constraint.append(a_rl_re + cp_x <= w_ub_sign)
            tmp = [var + (a_rl_re + cp_x).T @ daily_return_ay + z_i >= 0]
            cp_constraint += tmp
            cp_constraint.append(cp.SOC(socp_d, cov_sqrt_t1 @ (a_rl_re + cp_x)))
        # cvxpy
        while cnt <= cnt_th:
            try:
                obj_f2 = gamma * cp.quad_form(cp_x, cov_r_t1) + (1 - gamma) * (var + 1 / ((1 - 0.05)) * z_i)
                cp_obj = cp.Minimize(obj_f2)
                cp_prob = cp.Problem(cp_obj, cp_constraint)
                cp_prob.solve(solver=cp.ECOS, verbose=False)

                if cp_prob.status == 'optimal':
                    solver_flag = True
                    break
                else:
                    raise
            except:
                solver_flag = False
                cnt += 1
                risk_safe_t1 = risk_safe_t1 + step_add_lst[cnt - 2] + 0.01
                socp_d = -risk_market_t1 + risk_safe_t1 + (gamma - 1) * last_h_risk
                if a_rl.shape[0] != N:
                    cp_constraint[-1] = cp.SOC(socp_d, cov_sqrt_t1 @ (a_rl_re[1:] + cp_x[1:]))
                else:
                    cp_constraint[-1] = cp.SOC(socp_d, cov_sqrt_t1 @ (a_rl_re + cp_x))

        if (cp_prob.status == 'optimal') and solver_flag:
            is_solvable_status = True
            a_cbf = np.reshape(np.array(cp_x.value), -1)
            env.solver_stat['solvable'] = env.solver_stat['solvable'] + 1
            # Check the solution whether satisfy the risk constraint.
            env.risk_adj_lst[-1] = risk_safe_t1
            if a_rl.shape[0] != N:
                cur_alpha_risk = np.sqrt(
                    np.matmul(np.matmul((a_rl[1:] + a_cbf[1:]), cov_r_t1), (a_rl[1:] + a_cbf[1:]).T))
            else:
                cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl + a_cbf), cov_r_t1), (a_rl + a_cbf).T))
            assert (cur_alpha_risk - socp_d) <= 0.00001, 'cur risk: {}, socp_d {}'.format(cur_alpha_risk, socp_d)
            assert np.abs(np.sum(np.abs(a_rl + a_cbf)) - 1) <= 0.00001, 'sum of actions: {} \n{} \n{}'.format(
                np.sum(np.abs((a_rl + a_cbf))), a_rl, a_cbf)
            env.solvable_flag.append(0)
        else:
            if a_rl.shape[0] != N:
                a_cbf = np.zeros(N + 1)
                cur_alpha_risk = np.sqrt(
                    np.matmul(np.matmul((a_rl[1:] + a_cbf[1:]), cov_r_t1), (a_rl[1:] + a_cbf[1:]).T))
            else:
                a_cbf = np.zeros(N)
                cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl), cov_r_t1), (a_rl).T))

            env.solver_stat['insolvable'] = env.solver_stat['insolvable'] + 1
            is_solvable_status = False
            env.solvable_flag.append(1)
            # cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl), cov_r_t1), (a_rl).T))
            env.risk_adj_lst[-1] = risk_safe_t1
    env.risk_pred_lst.append(cur_alpha_risk)
    env.is_last_ctrl_solvable = is_solvable_status
    if not is_solvable_status:
        print(cur_date, "无解")
    if cnt > 1:
        env.stepcount = env.stepcount + 1
    # 求解器解出的action
    return a_cbf, is_solvable_status


def mask_gmv_cbf_opt(env, a_rl, pred_dict, mask):
    """
    The risk constraint is based on GMV-CVaR method. Not just considering the satisfaction of the current risk constraint, but also considering the trends/gradients of the future risk.
    """
    # pred_prices_change = pred_dict['shortterm']
    # pred_prices_change = pred_dict['shortterm']
    # cur_date_1 = str(env.curData['date'].unique()[0]).split('T')[0]
    a_rl = np.array(a_rl)
    assert np.sum(
        np.abs(a_rl)) - 1 < 0.0001, "The sum of a_rl is not equal to 1, the value is {}, a_rl list: {}".format(
        np.sum(a_rl), a_rl)
    N = env.config.topK
    # Past N days daily return rate of each stock, (num_of_stocks, lookback_days), [[t-N+1, t-N+2, .., t-1, t]]
    daily_return_ay = env.ctl_state['DAILYRETURNS-{}'.format(env.config.dailyRetun_lookback)]
    daily_return_ay = np.array([np.fromstring(row.strip("[]"), sep=" ") for row in daily_return_ay])
    daily_return_ay = np.mean(daily_return_ay, axis=1)
    day = 1
    while True:
        try:
            cur_date = env.curData['date'].unique()[0]
            cur_date_0 = str(np.datetime64(cur_date) - np.timedelta64(day, 'D')).split('T')[0]
            # cov_r_t0 = realized_covariance(cur_date_0)
            cov_r_t0 = np.loadtxt(f"/mnt/f/data/rcov/{cur_date_0}.txt")
            break
        except:
            # print("当前日期无数据,调整到上一日")
            day += 1
            cur_date_0 = str(np.datetime64(cur_date) - np.timedelta64(day, 'D')).split('T')[0]
    # cov_r_t0 = np.cov(daily_return_ay)
    if a_rl.shape[0] != N:
        mask = np.hstack(([[0]], mask))
        w_t0 = (np.array([env.actions_memory[-1]]))[:, 1:]
    else:
        mask = np.array(mask)
        w_t0 = np.array([env.actions_memory[-1]])
    try:
        risk_stg_t0 = np.sqrt(np.matmul(np.matmul(w_t0, cov_r_t0), w_t0.T)[0][0])
    except Exception as error:
        print("Risk-(MV model variance): {}".format(error))
        risk_stg_t0 = 0
    risk_market_t0 = env.config.risk_market
    if len(env.risk_adj_lst) <= 1:
        risk_safe_t0 = env.risk_adj_lst[-1]
    else:
        if env.is_last_ctrl_solvable:
            risk_safe_t0 = env.risk_adj_lst[-2]
        else:
            risk_safe_t0 = risk_stg_t0 + risk_market_t0

    gamma = env.config.cbf_gamma
    risk_market_t1 = env.config.risk_market
    risk_safe_t1 = env.risk_adj_lst[-1]

    cov_r_t1 = np.loadtxt(f"/mnt/f/data/rcov/{cur_date}.txt")
    cov_sqrt_t1 = sqrtm(cov_r_t1)
    cov_sqrt_t1 = cov_sqrt_t1.real
    cur_date = env.curData['date'].unique()[0]
    # return_entry_t1 = pd.read_csv(f"/mnt/f/data/difference_entry_rolling/{cur_date}.csv")['rolling_entry_5'].values.T
    # price_entry_t1 = pd.read_csv(f"/mnt/f/data/difference_entry_rolling/{cur_date}.csv")['rolling_difference_entry_5'].values.T

    # use_cvxopt_threshold = 10  # using cvxopt tool will be faster when the size of portfolio is less than or equal to 10. Otherwise, the cvxpy tool will be faster.
    w_lb = 0
    w_ub = 1

    a_rl_re_sign = np.reshape(a_rl, (-1, 1))
    sign_mul = np.ones((1, N + 1 if a_rl.shape[0] != N else N))
    w_lb_sign = w_lb
    w_ub_sign = w_ub

    last_h_risk = (-risk_market_t0 - risk_stg_t0 + risk_safe_t0)
    last_h_risk = np.max([last_h_risk, 0.0])
    socp_d = (-risk_market_t1 + risk_safe_t1 + (gamma - 1) * last_h_risk)  # 指定最小风险
    step_add_lst = [0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.05, 0.05]
    cnt = 1
    if env.config.is_enable_dynamic_risk_bound:
        cnt_th = env.config.ars_trial  # Iterative risk relaxation
    else:
        cnt_th = 1

    if True:
        # Complete solver
        # ++ Implemented by cvxpy
        if a_rl.shape[0] != N:
            cp_x = cp.Variable((N + 1, 1))
            a_rl_re = np.reshape(a_rl, (-1, 1))
            cp_constraint = []
            num_count = np.count_nonzero(mask == 0)
            weight_safe = 1 / num_count
            mask = (1 - mask)
            cp_constraint.append(a_rl_re[0] + cp_x[0] <= weight_safe)
            # cp_constraint.append(cp_x >= -0.1)
            cp_constraint.append(cp.sum((sign_mul * mask) @ cp_x) + cp.sum(a_rl_re_sign) == 1)
            cp_constraint.append(a_rl_re + cp.multiply(mask.T, cp_x) >= w_lb_sign)
            cp_constraint.append(a_rl_re + cp.multiply(mask.T, cp_x) <= w_ub_sign)
            cp_constraint.append(cp_x == cp.multiply(mask.T, cp_x))
            # cp_constraint.append(daily_return_ay.T @ cp.multiply(mask.T,cp_x)[1:]>=0)
            cp_constraint.append(cp.SOC(socp_d, cov_sqrt_t1 @ (a_rl_re[1:] + cp.multiply(mask.T, cp_x)[1:])))
        else:
            cp_x = cp.Variable((N, 1))
            a_rl_re = np.reshape(a_rl, (-1, 1))
            cp_constraint = []
            num_count = np.count_nonzero(mask == 0)
            weight_safe = 1 / num_count
            mask = (1 - mask)
            cp_constraint.append(a_rl_re + cp_x <= 0.1)
            # cp_constraint.append(cp_x >= -0.1)
            cp_constraint.append(cp.sum((sign_mul * mask) @ cp_x) + cp.sum(a_rl_re_sign) == 1)
            cp_constraint.append(a_rl_re + cp.multiply(mask.T, cp_x) >= w_lb_sign)
            cp_constraint.append(a_rl_re + cp.multiply(mask.T, cp_x) <= w_ub_sign)
            cp_constraint.append(cp_x == cp.multiply(mask.T, cp_x))
            # cp_constraint.append(daily_return_ay.T @ cp.multiply(mask.T,cp_x)>=0)
            cp_constraint.append(cp.SOC(socp_d, cov_sqrt_t1 @ (a_rl_re + cp.multiply(mask.T, cp_x))))
        # cvxpy
        while cnt <= cnt_th:
            try:
                # price_entropy = cp_x[1:].T @ price_entry_t1
                obj_f2 = cp.sum_squares(cp_x)
                # obj_f2 = (cp.multiply(mask.T, cp_x)[1:]).T @ env.cur_hidden_vector_ay[-1].T
                cp_obj = cp.Minimize(obj_f2)
                cp_prob = cp.Problem(cp_obj, cp_constraint)
                cp_prob.solve(solver=cp.ECOS, verbose=False)

                if cp_prob.status == 'optimal':
                    solver_flag = True
                    break
                else:
                    raise
            except Exception as e:
                # print(e)
                solver_flag = False
                cnt += 1
                risk_safe_t1 = risk_safe_t1 + step_add_lst[cnt - 2]
                socp_d = (-risk_market_t1 + risk_safe_t1 + (gamma - 1) * last_h_risk)
                # socp_d += 0.1
                if a_rl.shape[0] != N:
                    cp_constraint[-1] = cp.SOC(socp_d, cov_sqrt_t1 @ (a_rl_re[1:] + cp.multiply(mask.T, cp_x)[1:]))
                else:
                    cp_constraint[-1] = cp.SOC(socp_d, cov_sqrt_t1 @ (a_rl_re + cp.multiply(mask.T, cp_x)))

        if (cp_prob.status == 'optimal') and solver_flag:
            is_solvable_status = True
            a_cbf = np.reshape(np.array(cp_x.value), -1)
            env.solver_stat['solvable'] = env.solver_stat['solvable'] + 1
            # Check the solution whether satisfy the risk constraint.
            env.risk_adj_lst[-1] = risk_safe_t1
            # 策略风险
            if a_rl.shape[0] != N:
                cur_alpha_risk = np.sqrt(
                    np.matmul(np.matmul((a_rl[1:] + a_cbf[1:]), cov_r_t1), (a_rl[1:] + a_cbf[1:]).T))
            else:
                cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl + a_cbf), cov_r_t1), (a_rl + a_cbf).T))
            assert (cur_alpha_risk - socp_d) <= 0.00001, 'cur risk: {}, socp_d {}'.format(cur_alpha_risk, socp_d)
            assert np.abs(np.sum(np.abs(a_rl + a_cbf)) - 1) <= 0.00001, 'sum of actions: {} \n{} \n{}'.format(
                np.sum(np.abs((a_rl + a_cbf))), a_rl, a_cbf)
            env.solvable_flag.append(0)
        else:
            if a_rl.shape[0] != N:
                a_cbf = np.zeros(N + 1)
                cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl[1:]), cov_r_t1), (a_rl[1:]).T))
            else:
                a_cbf = np.zeros(N)
                cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl), cov_r_t1), (a_rl).T))

            env.solver_stat['insolvable'] = env.solver_stat['insolvable'] + 1
            is_solvable_status = False
            env.solvable_flag.append(1)
            env.risk_adj_lst[-1] = risk_safe_t1
    env.risk_pred_lst.append(cur_alpha_risk)
    env.is_last_ctrl_solvable = is_solvable_status
    if not is_solvable_status:
        print(cur_date, "无解")
    if cnt > 1:
        env.stepcount = env.stepcount + 1
    # 求解器解出的action
    return a_cbf, is_solvable_status


def mask_cvar_gmv_cbf_opt(env, a_rl, pred_dict, mask):
    """
    The risk constraint is based on GMV-CVaR method. Not just considering the satisfaction of the current risk constraint, but also considering the trends/gradients of the future risk.
    """
    # pred_prices_change = pred_dict['shortterm']
    cur_date_1 = str(env.curData['date'].unique()[0]).split('T')[0]
    a_rl = np.array(a_rl)
    assert np.sum(
        np.abs(a_rl)) - 1 < 0.0001, "The sum of a_rl is not equal to 1, the value is {}, a_rl list: {}".format(
        np.sum(a_rl), a_rl)
    N = env.config.topK
    # Past N days daily return rate of each stock, (num_of_stocks, lookback_days), [[t-N+1, t-N+2, .., t-1, t]]
    daily_return_ay = env.ctl_state['DAILYRETURNS-{}'.format(env.config.dailyRetun_lookback)]
    daily_return_ay = np.array([np.fromstring(row.strip("[]"), sep=" ") for row in daily_return_ay])
    # cov_r_t0 = realized_covariance(cur_date_1)
    cov_r_t0 = np.loadtxt(f"/mnt/f/data/rcov/{cur_date_1}.txt")
    var_r_t0 = np.diag(cov_r_t0)
    # cov_r_t0 = np.cov(daily_return_ay)
    if a_rl.shape[0] != N:
        mask = np.hstack(([[0]], mask))
        w_t0 = (np.array([env.actions_memory[-1]]) * (1 - mask))[:, 1:]
    else:
        mask = np.array(mask)
        w_t0 = np.array([env.actions_memory[-1]]) * (1 - mask)
    try:
        risk_stg_t0 = np.sqrt(np.matmul(np.matmul(w_t0, cov_r_t0), w_t0.T)[0][0])
    except Exception as error:
        print("Risk-(MV model variance): {}".format(error))
        risk_stg_t0 = 0
    risk_market_t0 = env.config.risk_market
    if len(env.risk_adj_lst) <= 1:
        risk_safe_t0 = env.risk_adj_lst[-1]
    else:
        if env.is_last_ctrl_solvable:
            risk_safe_t0 = env.risk_adj_lst[-2]
        else:
            risk_safe_t0 = risk_stg_t0 + risk_market_t0

    gamma = env.config.cbf_gamma
    risk_market_t1 = env.config.risk_market
    risk_safe_t1 = env.risk_adj_lst[-1]
    # cov_r_t1 = realized_covariance(cur_date_1)
    # # lw = LedoitWolf()
    # # cov_r_t1 = lw.fit(cov_r_t1).covariance_
    # # cov_r_t1 = OAS().fit(cov_r_t1).covariance_
    # cov_sqrt_t1 = sqrtm(cov_r_t1)
    # cov_sqrt_t1 = cov_sqrt_t1.real
    daily_return_ay = np.mean(daily_return_ay, axis=1)
    # eigenvalues = np.linalg.eigvals(cov_r_t1)
    # is_psd = np.all(eigenvalues >= 0)
    # if ~is_psd:
    #     print('协方差阵不是半正定的')
    w_lb = 0
    w_ub = 1
    last_h_risk = (-risk_market_t0 - risk_stg_t0 + risk_safe_t0)
    last_h_risk = np.max([last_h_risk, 0.0])
    a_rl_re_sign = np.reshape(a_rl, (-1, 1))
    sign_mul = np.ones((1, N + 1 if a_rl.shape[0] != N else N))
    w_lb_sign = w_lb
    w_ub_sign = w_ub
    socp_d = -risk_market_t1 + risk_safe_t1 + (gamma - 1) * last_h_risk  # 指定最小风险

    step_add_lst = [0.002, 0.002, 0.002, 0.002, 0.002, 0.005, 0.005, 0.005, 0.008, 0.01]
    cnt = 1
    if True:
        # Complete solver
        # ++ Implemented by cvxpy
        if len(env.var_lst) < 5:
            var = env.var_lst[-1]
        else:
            var = np.mean(env.var_lst[-5:])
        if a_rl.shape[0] != N:
            cp_x = cp.Variable((N + 1, 1))
            z_i = cp.Variable((1, 1), nonneg=True)  # J 维辅助变量 z
            a_rl_re = np.reshape(a_rl, (-1, 1))
            cp_constraint = []
            mask = (1 - mask)
            cp_constraint.append(a_rl_re + cp_x <= 0.1)
            # cp_constraint.append(a_rl_re[1:]+cp_x[1:]<=0.5)
            cp_constraint.append(cp.sum((sign_mul * mask) @ cp_x) + cp.sum(a_rl_re_sign) == 1)
            cp_constraint.append(a_rl_re + cp.multiply(mask.T, cp_x) >= w_lb_sign)
            cp_constraint.append(a_rl_re + cp.multiply(mask.T, cp_x) <= w_ub_sign)
            cp_constraint.append(cp_x == cp.multiply(mask.T, cp_x))
            # cp_constraint.append(cp.sum(cp.abs((a_rl_re[1:] +cp.multiply(mask.T,cp_x)[1:]) - env.actions_memory[-1][1:].reshape(-1, 1))) <= 0.2)
            tmp = [var + (a_rl_re[1:] + cp.multiply(mask.T, cp_x)[1:]).T @ daily_return_ay + z_i >= 0]
            cp_constraint += tmp

            # obj_f1 = t1_entry@cp_x[1:]
            # cp_constraint.append(var + 1/(5*(1-0.05)) * z_i <= socp_d)
            # cp_constraint.append(cp.SOC(socp_d, cov_sqrt_t1 @ (a_rl_re[1:] + cp.multiply(mask.T,cp_x)[1:])))
        else:
            cp_x = cp.Variable((N, 1))
            z_i = cp.Variable((1, 1), nonneg=True)  # J 维辅助变量 z
            a_rl_re = np.reshape(a_rl, (-1, 1))
            cp_constraint = []
            mask = (1 - mask)
            cp_constraint.append(a_rl_re + cp_x <= 0.1)
            # cp_constraint.append(a_rl_re[1:]+cp_x[1:]<=0.5)
            cp_constraint.append(cp.sum((sign_mul * mask) @ cp_x) + cp.sum(a_rl_re_sign) == 1)
            cp_constraint.append(a_rl_re + cp.multiply(mask.T, cp_x) >= w_lb_sign)
            cp_constraint.append(a_rl_re + cp.multiply(mask.T, cp_x) <= w_ub_sign)
            cp_constraint.append(cp_x == cp.multiply(mask.T, cp_x))
            # cp_constraint.append(cp.sum(cp.abs((a_rl_re +cp.multiply(mask.T,cp_x)) - env.actions_memory[-1].reshape(-1,1))) <= 0.2)
            tmp = [var + (a_rl_re + cp.multiply(mask.T, cp_x)).T @ daily_return_ay + z_i >= 0]
            cp_constraint += tmp
        # cvxpy
        while cnt <= 10:
            try:
                if a_rl.shape[0] != N:
                    obj_f2 = (var + (1 / (1 - 0.05)) * z_i)
                else:
                    obj_f2 = (var + (1 / (1 - 0.05)) * z_i)
                cp_obj = cp.Minimize(obj_f2)
                cp_prob = cp.Problem(cp_obj, cp_constraint)
                cp_prob.solve(solver=cp.ECOS, verbose=False)

                if cp_prob.status == 'optimal':
                    solver_flag = True
                    break
                else:
                    raise
            except:
                solver_flag = False
                cnt += 1
                risk_safe_t1 = risk_safe_t1 + step_add_lst[cnt - 2] + 0.01
                socp_d = (-risk_market_t1 + risk_safe_t1 + (gamma - 1) * last_h_risk)
                # socp_d += 0.1
                if a_rl.shape[0] != N:
                    cp_constraint[0] = a_rl_re + cp_x <= 0.1
                else:
                    cp_constraint[0] = a_rl_re + cp_x <= 0.1

        if (cp_prob.status == 'optimal') and solver_flag:
            is_solvable_status = True
            a_cbf = np.reshape(np.array(cp_x.value), -1)
            env.solver_stat['solvable'] = env.solver_stat['solvable'] + 1
            # Check the solution whether satisfy the risk constraint.
            env.risk_adj_lst[-1] = risk_safe_t1
            cur_alpha_risk = cp_prob.value
            # assert (cur_alpha_risk - socp_d) <= 0.00001, 'cur risk: {}, socp_d {}'.format(cur_alpha_risk, socp_d)
            assert np.abs(np.sum(np.abs(a_rl + a_cbf)) - 1) <= 0.00001, 'sum of actions: {} \n{} \n{}'.format(
                np.sum(np.abs((a_rl + a_cbf))), a_rl, a_cbf)
            env.solvable_flag.append(0)
        else:
            if a_rl.shape[0] != N:
                a_cbf = np.zeros(N + 1)
                cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl[1:]), cov_r_t0), (a_rl[1:]).T))
            else:
                a_cbf = np.zeros(N)
                cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl), cov_r_t0), (a_rl).T))
            env.solver_stat['insolvable'] = env.solver_stat['insolvable'] + 1
            is_solvable_status = False
            env.solvable_flag.append(1)
            # cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl), cov_r_t1), (a_rl).T))
            env.risk_adj_lst[-1] = risk_safe_t1
    env.risk_pred_lst.append(cur_alpha_risk)
    env.is_last_ctrl_solvable = is_solvable_status
    if not is_solvable_status:
        print(cur_date_1, "无解")
    if cnt > 1:
        env.stepcount = env.stepcount + 1
    # 求解器解出的action
    return a_cbf, is_solvable_status


def mask_cvar_cbf_opt(env, a_rl, mask):
    """
    The risk constraint is based on GMV-CVaR method. Not just considering the satisfaction of the current risk constraint, but also considering the trends/gradients of the future risk.
    """
    # pred_prices_change = pred_dict['shortterm']
    cur_date_1 = str(env.curData['date'].unique()[0]).split('T')[0]
    a_rl = np.array(a_rl)
    assert np.sum(
        np.abs(a_rl)) - 1 < 0.0001, "The sum of a_rl is not equal to 1, the value is {}, a_rl list: {}".format(
        np.sum(a_rl), a_rl)
    N = env.config.topK
    risk_safe_t1 = env.risk_adj_lst[-1]
    # cov_r_t1 = np.cov(r_t1)
    # cov_r_t1 = realized_covariance(cur_date_1)
    daily_return_ay_last = env.curData['ret_{}'.format(env.config.dailyRetun_lookback)].values
    daily_return_ay = np.array([np.array(row) for row in daily_return_ay_last])

    # daily_return_ay = np.array([np.fromstring(row.strip("[]"), sep=" ") for row in daily_return_ay])
    cov_r_t1 = np.cov(daily_return_ay)
    # cov_r_t1 = np.loadtxt(f"/mnt/f/data/rcov/{cur_date_1}.txt")
    # cov_sqrt_t1 = sqrtm(cov_r_t1)
    # cov_sqrt_t1 = cov_sqrt_t1.real
    # daily_return_ay = daily_return_ay[:,-1] + np.diag(cov_sqrt_t1)
    # daily_return_ay = env.ctl_state['DAILYRETURNS-{}'.format(env.config.dailyRetun_lookback)]
    # daily_return_ay = np.array([np.fromstring(row.strip("[]"), sep=" ") for row in daily_return_ay])
    # daily_return_ay = np.mean(daily_return_ay, axis=1)
    # daily_return_ay = env.cur_hidden_vector_ay[-1]
    w_lb = 0
    w_ub = 1
    # num_count = np.count_nonzero(mask == 0)
    # risk_safe_return_t1 = 1/num_count
    a_rl_re_sign = np.reshape(a_rl, (-1, 1))
    sign_mul = np.ones((1, N + 1 if a_rl.shape[0] != N else N))
    w_lb_sign = w_lb
    w_ub_sign = w_ub
    # socp_d = -risk_market_t1 + risk_safe_t1 + (gamma - 1) * last_h_risk  # 指定最小风险
    step_add_lst = [0.002, 0.004, 0.006, 0.008, 0.008, 0.01, 0.01, 0.01, 0.015, 0.02]
    cnt = 1
    # 分别估计每支股票的均值和标准差
    mean_returns = np.mean(daily_return_ay, axis=0)  # (num_assets,)
    std_returns = np.std(daily_return_ay, axis=0, ddof=1)  # (num_assets,)

    # 生成未来收益：t分布随机变量 * 标准差 + 均值
    simulated_z = t.rvs(df=3, size=(N, 50))

    # Step 3: 广播计算模拟收益：每行资产 × 100 次
    simulated_returns = mean_returns[:, None] + std_returns[:, None] * simulated_z
    
    if True:
        # Complete solver
        # ++ Implemented by cvxpy
        if len(env.var_lst) < 5:
            var = env.var_lst[-1]
        else:
            var = np.mean(env.var_lst[-5:])
        # var = env.var_lst[-1]
        if a_rl.shape[0] != N:
            mask = np.hstack(([[0]], mask))
            # a = mask_cvar_cbf_opt_1(env, a_rl, pred_dict, mask)
            num_count = np.count_nonzero(mask == 0)
            #risk_safe_return_t1 = 1 / (10*num_count)
            #if env.opt<=30:
            #    risk_safe_return_t1 = 1 / (num_count)
            #elif env.opt>30:
            risk_safe_return_t1 = 1 / (10*num_count)
            #else:
            #    risk_safe_return_t1 = 1 / (50*num_count)
            weight_safe = min(a_rl[0], risk_safe_return_t1)
            cp_x = cp.Variable((N + 1, 1))
            z_i = cp.Variable((1, 1), nonneg=True)  # J 维辅助变量 z
            a_rl_re = np.reshape(a_rl, (-1, 1))
            cp_constraint = []
            mask = (1 - mask)
            cp_constraint.append(a_rl_re[0] + cp_x[0] <= weight_safe)
            # cp_constraint.append(a_rl_re[1:]+cp_x[1:]<=0.5)
            cp_constraint.append(cp.sum((sign_mul * mask) @ cp_x) + cp.sum(a_rl_re_sign) == 1)
            cp_constraint.append(a_rl_re + cp.multiply(mask.T, cp_x) >= w_lb_sign)
            cp_constraint.append(a_rl_re + cp.multiply(mask.T, cp_x) <= w_ub_sign)
            cp_constraint.append(cp_x == cp.multiply(mask.T, cp_x))
            # cp_constraint.append(cp.sum(cp.abs((a_rl_re[1:] +cp.multiply(mask.T,cp_x)[1:]) - env.actions_memory[-1][1:].reshape(-1, 1))) <= 0.2)
            #tmp = [var + (a_rl_re[1:] + cp.multiply(mask.T, cp_x)[1:]).T @ env.cur_hidden_vector_ay[-1].T + z_i[0] >= 0]
            #tmp += [var + (a_rl_re[1:] + cp.multiply(mask.T, cp_x)[1:]).T @ daily_return_ay[:,-3:].mean(axis=1) + z_i[1] >= 0]
            #tmp = [var + (a_rl_re[1:] + cp.multiply(mask.T, cp_x)[1:]).T @ daily_return_ay[:,-i] + z_i[i] >= 0 for i in range(1,3)]
            #cp_constraint += tmp
            # obj_f1 = t1_entry@cp_x[1:]
            #cp_constraint.append(var + 1/(10*(1-0.05)) * cp.sum(z_i) <= 1)
            # cp_constraint.append(cp.SOC(socp_d, cov_sqrt_t1 @ (a_rl_re[1:] + cp.multiply(mask.T,cp_x)[1:])))
        else:
            cp_x = cp.Variable((N, 1))
            z_i = cp.Variable((1, 1), nonneg=True)  # J 维辅助变量 z
            a_rl_re = np.reshape(a_rl, (-1, 1))
            cp_constraint = []
            mask = (1 - mask)
            num_count = np.count_nonzero(mask == 0)
            weight_safe = 1 / num_count
            cp_constraint.append(a_rl_re + cp_x == a_rl[0])
            # cp_constraint.append(a_rl_re[1:]+cp_x[1:]<=0.5)
            cp_constraint.append(cp.sum((sign_mul * mask) @ cp_x) + cp.sum(a_rl_re_sign) == 1)
            cp_constraint.append(a_rl_re + cp.multiply(mask.T, cp_x) >= w_lb_sign)
            cp_constraint.append(a_rl_re + cp.multiply(mask.T, cp_x) <= w_ub_sign)
            cp_constraint.append(cp_x == cp.multiply(mask.T, cp_x))
            # cp_constraint.append(cp.sum(cp.abs((a_rl_re +cp.multiply(mask.T,cp_x)) - env.actions_memory[-1].reshape(-1,1))) <= 0.2)
            #tmp = [var + (a_rl_re + cp.multiply(mask.T, cp_x)).T @ env.cur_hidden_vector_ay[-1].T + z_i >= 0]
            #cp_constraint += tmp
            # cp_constraint.append(var + 1 / (5 * (1 - 0.05)) * z_i <= 0.05)
        # cvxpy
        try:
            if a_rl.shape[0] != N:
                obj_f2 = (var + (1 / (1*(1 - 0.05))) * cp.sum(z_i))
                obj_f1 = cp.sum_squares(cp_x)
                obj_f3 = cp.sum(cp.abs(cp_x))
                # obj_f2 = (cp.multiply(mask.T, cp_x)[1:]).T @ env.cur_hidden_vector_ay[-1].T
            else:
                obj_f2 = (var + (1 / (1 - 0.05)) * z_i)
                obj_f1 = cp.sum_squares(cp_x)
                obj_f3 = cp.sum(cp.abs(cp_x))
            cp_obj = cp.Minimize(obj_f2+0.00001*obj_f1)
            # cp_obj = cp.Minimize(obj_f2+obj_f1*0.0000001)
            cp_prob = cp.Problem(cp_obj, cp_constraint)
            cp_prob.solve(solver=cp.ECOS, verbose=False)

            if cp_prob.status == 'optimal':
                solver_flag = True
            else:
                raise
        except:
            solver_flag = False
            cnt += 1
            risk_safe_t1 = risk_safe_t1 + step_add_lst[cnt -2] + 0.01
        if (cp_prob.status == 'optimal') and solver_flag:
            is_solvable_status = True
            a_cbf = np.reshape(np.array(cp_x.value), -1)
            env.solver_stat['solvable'] = env.solver_stat['solvable'] + 1
            # Check the solution whether satisfy the risk constraint.
            env.risk_adj_lst[-1] = risk_safe_t1
            cur_alpha_risk = cp_prob.value
            # assert (cur_alpha_risk - socp_d) <= 0.00001, 'cur risk: {}, socp_d {}'.format(cur_alpha_risk, socp_d)
            assert np.abs(np.sum(np.abs(a_rl + a_cbf)) - 1) <= 0.00001, 'sum of actions: {} \n{} \n{}'.format(
                np.sum(np.abs((a_rl + a_cbf))), a_rl, a_cbf)
            env.solvable_flag.append(0)
        else:
            if a_rl.shape[0] != N:
                a_cbf = np.zeros(N + 1)
                cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl[1:]), cov_r_t1), (a_rl[1:]).T))
            else:
                a_cbf = np.zeros(N)
                cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl), cov_r_t1), (a_rl).T))
            env.solver_stat['insolvable'] = env.solver_stat['insolvable'] + 1
            is_solvable_status = False
            env.solvable_flag.append(1)
            # cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl), cov_r_t1), (a_rl).T))
            env.risk_adj_lst[-1] = risk_safe_t1
    env.risk_pred_lst.append(cur_alpha_risk)
    env.is_last_ctrl_solvable = is_solvable_status
    if not is_solvable_status:
        print(cur_date_1, "无解")
    if cnt > 1:
        env.stepcount = env.stepcount + 1
    # 求解器解出的action
    # b=a_rl+a_cbf

    return a_cbf, is_solvable_status


def mask_entry_opt(env, a_rl, pred_dict, mask):
    """
    The risk constraint is based on controller barrier function (CBF) method. Not just considering the satisfaction of the current risk constraint, but also considering the trends/gradients of the future risk.
    """
    # pred_prices_change = pred_dict['shortterm']
    # cur_date_1 = str(env.curData['date'].unique()[0]).split('T')[0]
    # import pdb
    # pdb.set_trace()
    a_rl = np.array(a_rl)
    assert np.sum(
        np.abs(a_rl)) - 1 < 0.0001, "The sum of a_rl is not equal to 1, the value is {}, a_rl list: {}".format(
        np.sum(a_rl), a_rl)
    N = env.config.topK
    day = 1
    while True:
        try:
            cur_date = env.curData['date'].unique()[0]
            cur_date_0 = str(np.datetime64(cur_date) - np.timedelta64(day, 'D')).split('T')[0]
            if env.config.if_crp:
                cov_r_t0 = np.loadtxt(f"/kaggle/input/cspara/datasets/mi_crp_entry_True_cov_lw/{cur_date_0}.txt")
            else:
                cov_r_t0 = np.loadtxt(f"/kaggle/input/cspara/datasets/mi_entry_cov_lw/{cur_date_0}.txt")
            # return_entry_t0 = pd.read_csv(f"/mnt/f/data/difference_entry_rolling/{cur_date_0}.csv")['rolling_entry_5'].values.T
            # price_entry_t0 = pd.read_csv(f"/mnt/f/data/difference_entry_rolling/{cur_date_0}.csv")['rolling_difference_entry_5'].values.T
            break
        except:
            # print("当前日期无数据,调整到上一日")
            day += 1
            # cur_date_0 = str(np.datetime64(cur_date)  - np.timedelta64(day, 'D')).split('T')[0]
    if a_rl.shape[0] != N:
        mask = np.hstack(([[0]], mask))
        w_t0 = (np.array([env.actions_memory[-1]]))[:, 1:]
    else:
        mask = np.array(mask)
        w_t0 = np.array([env.actions_memory[-1]])
    try:
        risk_stg_return_t0 = np.sqrt(np.matmul(np.matmul(w_t0, cov_r_t0), w_t0.T)[0][0])
        # risk_stg_return_t0 = np.sqrt(np.matmul(np.matmul((a_rl[1:]), cov_r_t0), (a_rl[1:]).T))
        # risk_stg_price_t0 = (w_t0 @ price_entry_t0)[0]
    except Exception as error:
        print("Risk-(MV model variance): {}".format(error))
        risk_stg_return_t0 = 0
    # risk_market_t0 = env.config.risk_market
    # if len(env.risk_adj_lst) <= 1:
    #     risk_safe_return_t0 = env.risk_adj_lst[-1]
    #     risk_safe_price_t0 = env.risk_adj_lst[-1]
    # else:
    #     if env.is_last_ctrl_solvable:
    #         risk_safe_return_t0 = env.risk_adj_lst[-2]
    #         risk_safe_price_t0 = env.risk_adj_lst[-2]
    #     else:
    #         risk_safe_return_t0 = risk_stg_return_t0 + risk_market_t0
    #         # risk_safe_price_t0 = risk_stg_price_t0 + risk_market_t0

    num_count = np.count_nonzero(mask == 0)
    risk_safe_return_t1 = 1 / (10 * num_count)
    weight_safe = min(a_rl[0], risk_safe_return_t1)
    gamma = env.config.cbf_gamma
    risk_market_t1 = env.config.risk_market
    # risk_safe_return_t1 = env.risk_adj_lst[-1]
    # risk_safe_price_t1 = env.risk_adj_lst[-1]

    if env.config.if_crp:
        cov_r_t1 = np.loadtxt(f"/kaggle/input/cspara/datasets/mi_crp_entry_True_cov_lw/{cur_date}.txt")
    else:
        cov_r_t1 = np.loadtxt(f"/kaggle/input/cspara/datasets/mi_entry_cov_lw/{cur_date}.txt")
    # cov_r_t1, shrinkage_coef = ledoit_wolf_shrinkage(cov_r_t1,assume_centered=True)
    # cov_r_t1 = ledoit_wolf_cov(cov_r_t1_1)
    cov_sqrt_t1 = sqrtm(cov_r_t1)
    cov_sqrt_t1 = cov_sqrt_t1.real
    # return_entry_t1 = pd.read_csv(f"/mnt/f/data/difference_entry_rolling/{cur_date}.csv")['rolling_entry_5'].values.T
    # price_entry_t1 = pd.read_csv(f"/mnt/f/data/difference_entry_rolling/{cur_date}.csv")['rolling_difference_entry_5'].values.T
    w_lb = 0
    w_ub = 1

    a_rl_re_sign = np.reshape(a_rl, (-1, 1))
    sign_mul = np.ones((1, N + 1 if a_rl.shape[0] != N else N))
    w_lb_sign = w_lb
    w_ub_sign = w_ub

    # last_h_return_risk = (-risk_market_t0 - risk_stg_return_t0 + risk_safe_return_t0)
    # # last_h_price_risk = (-risk_market_t0 - risk_stg_price_t0 + risk_safe_price_t0)
    # last_h_return_risk = np.max([last_h_return_risk, 0.0])
    # last_h_price_risk = np.max([last_h_price_risk, 0.0])

    socp_return_d = (-risk_market_t1 + gamma * risk_safe_return_t1 + (1 - gamma) * risk_stg_return_t0)  # 指定最小风险
    # socp_return_d = np.max([socp_return_d, 0.0])
    # socp_price_d = (-risk_market_t1 + risk_safe_price_t1 + (gamma - 1) * last_h_price_risk) # 指定最小风险
    # socp_return_d = 0.1
    step_add_return_lst = [0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 0.9]

    cnt = 1
    # if env.config.is_enable_dynamic_risk_bound:
    #     cnt_th = env.config.ars_trial  # Iterative risk relaxation
    # else:
    #     cnt_th = 1
    if True:
        # Complete solver
        # ++ Implemented by cvxpy
        if a_rl.shape[0] != N:
            cp_x = cp.Variable((N + 1, 1))
            a_rl_re = np.reshape(a_rl, (-1, 1))
            cp_constraint = []
            mask = (1 - mask)
            cp_constraint.append(a_rl_re[0] + cp_x[0] <= weight_safe)
            cp_constraint.append(cp.sum((sign_mul * mask) @ cp_x) + cp.sum(a_rl_re_sign) == 1)
            cp_constraint.append(a_rl_re + cp.multiply(mask.T, cp_x) >= w_lb_sign)
            cp_constraint.append(a_rl_re + cp.multiply(mask.T, cp_x) <= w_ub_sign)
            cp_constraint.append(cp_x == cp.multiply(mask.T, cp_x))
            cp_constraint.append(cp.SOC(socp_return_d, cov_sqrt_t1 @ (a_rl_re[1:] + cp.multiply(mask.T, cp_x)[1:])))
            # cp_constraint.append((a_rl_re[1:] +cp.multiply(mask.T,cp_x)[1:]).T @ return_entry_t1 >= socp_return_d)
            # cp_constraint.append((a_rl_re[1:] +cp.multiply(mask.T,cp_x)[1:]).T @ price_entry_t1 <= socp_price_d)
            # cp_constraint.append(cp.quad_form(a_rl_re[1:] +cp.multiply(mask.T,cp_x)[1:], cov_r_t1) <= socp_return_d)
        else:
            cp_x = cp.Variable((N, 1))
            a_rl_re = np.reshape(a_rl, (-1, 1))
            cp_constraint = []
            mask = (1 - mask)
            cp_constraint.append(a_rl_re + cp_x <= 0.1)
            cp_constraint.append(cp.sum((sign_mul * mask) @ cp_x) + cp.sum(a_rl_re_sign) == 1)
            cp_constraint.append(a_rl_re + cp.multiply(mask.T, cp_x) >= w_lb_sign)
            cp_constraint.append(a_rl_re + cp.multiply(mask.T, cp_x) <= w_ub_sign)
            cp_constraint.append(cp_x == cp.multiply(mask.T, cp_x))
            cp_constraint.append(cp.SOC(socp_return_d, cov_sqrt_t1 @ (a_rl_re + cp.multiply(mask.T, cp_x))))
            # cp_constraint.append((a_rl_re + cp.multiply(mask.T, cp_x)).T @ return_entry_t1 >= socp_return_d)
            # cp_constraint.append(cp.quad_form(a_rl_re + cp.multiply(mask.T, cp_x), cov_r_t1) <= socp_return_d)
            # cp_constraint.append((a_rl_re + cp.multiply(mask.T, cp_x)).T @ price_entry_t1 <= socp_price_d)
        # cvxpy
        while cnt <= 10:
            try:
                if a_rl.shape[0] != N:
                    # obj_f2 = cp.sum_squares(cp_x[1:]) cp_x[1:].T @ daily_return_ay_1.T -0.7*price_entropy - 0.3*return_entropy
                    # return_entropy = (cp.multiply(mask.T, cp_x)[1:]).T @ return_entry_t1  # 使用 log2 计算
                    # price_entropy = (cp.multiply(mask.T, cp_x)[1:]).T @ price_entry_t1
                    obj_f2 = cp.sum_squares(cp_x)
                    # obj_f2 = (cp.multiply(mask.T, cp_x)[1:]).T @ env.cur_hidden_vector_ay[-1]
                else:
                    # obj_f2 = cp.sum_squares(cp_x)
                    # return_entropy = (cp.multiply(mask.T, cp_x)).T @ return_entry_t1  # 使用 log2 计算
                    # price_entropy = (cp.multiply(mask.T, cp_x)).T @ price_entry_t1
                    obj_f2 = cp.sum_squares(cp_x)
                cp_obj = cp.Minimize(obj_f2)
                cp_prob = cp.Problem(cp_obj, cp_constraint)
                cp_prob.solve(solver=cp.ECOS, verbose=False)

                if cp_prob.status == 'optimal':
                    solver_flag = True
                    break
                else:
                    raise
            except Exception as e:
                # print(e)
                solver_flag = False
                cnt += 1
                risk_safe_return_t1 = risk_safe_return_t1 + step_add_return_lst[cnt - 2]
                # risk_safe_price_t1 = risk_safe_price_t1 + step_add_price_lst[cnt - 2]

                socp_return_d = (-risk_market_t1 + gamma * risk_safe_return_t1 + (1 - gamma) * risk_stg_return_t0)
                # socp_price_d = (-risk_market_t1 + risk_safe_price_t1 + (gamma - 1) * last_h_price_risk)
                # socp_return_d += 0.3
                # cp_constraint[0] = a_rl_re + cp_x <= (1/num_count)+socp_return_d
                cp_constraint[-1] = cp.SOC(socp_return_d, cov_sqrt_t1 @ (a_rl_re[1:] + cp.multiply(mask.T, cp_x)[1:]))
                # cp_constraint[-1] = (cp.quad_form((a_rl_re[1:] +cp.multiply(mask.T,cp_x)[1:]), cov_r_t1) <= socp_return_d**2)
                # cp_constraint[-1] = (a_rl_re[1:] +cp.multiply(mask.T,cp_x)[1:]).T @ return_entry_t1 >= socp_return_d
                # cp_constraint[-1] = (a_rl_re[1:] +cp.multiply(mask.T,cp_x)[1:]).T @ price_entry_t1 <= socp_price_d

        if (cp_prob.status == 'optimal') and solver_flag:
            is_solvable_status = True
            a_cbf = np.reshape(np.array(cp_x.value), -1)
            env.solver_stat['solvable'] = env.solver_stat['solvable'] + 1
            # Check the solution whether satisfy the risk constraint.
            # env.risk_adj_lst[-1] = risk_safe_return_t1
            # env.risk_adj_price_lst[-1] = risk_safe_price_t1
            # 策略风险
            cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl[1:] + a_cbf[1:]), cov_r_t1), (a_rl[1:] + a_cbf[1:]).T))
            # assert (cur_alpha_risk - socp_d) <= 0.00001, 'cur risk: {}, socp_d {}'.format(cur_alpha_risk, socp_d)
            assert np.abs(np.sum(np.abs(a_rl + a_cbf)) - 1) <= 0.00001, 'sum of actions: {} \n{} \n{}'.format(
                np.sum(np.abs((a_rl + a_cbf))), a_rl, a_cbf)
            env.solvable_flag.append(0)
        else:
            if a_rl.shape[0] != N:
                a_cbf = np.zeros(N + 1)
                cur_alpha_risk = np.sqrt(np.matmul(np.matmul((a_rl[1:]), cov_r_t1), (a_rl[1:]).T))
            else:
                a_cbf = np.zeros(N)
                # cur_alpha_risk = a_rl @ return_entry_t1 + a_rl @ price_entry_t1
                cur_alpha_risk = np.sqrt(np.matmul(np.matmul(a_rl, cov_r_t1), a_rl.T))

            env.solver_stat['insolvable'] = env.solver_stat['insolvable'] + 1
            is_solvable_status = False
            env.solvable_flag.append(1)
            # env.risk_adj_lst[-1] = risk_safe_return_t1
            # env.risk_adj_price_lst[-1] = risk_safe_price_t1

    env.risk_pred_lst.append(cur_alpha_risk)
    env.is_last_ctrl_solvable = is_solvable_status
    if not is_solvable_status:
        print(cur_date, "无解", cur_alpha_risk)
    if cnt > 1:
        env.stepcount = env.stepcount + 1
    # 求解器解出的action
    return a_cbf, is_solvable_status


def ledoit_wolf_cov(emp_cov):
    shrinkage = ledoit_wolf_shrinkage(
        emp_cov, assume_centered=True)
    n_features = emp_cov.shape[1]
    mu = np.sum(np.trace(emp_cov)) / n_features
    shrunk_cov = (1.0 - shrinkage) * emp_cov
    shrunk_cov.flat[:: n_features + 1] += shrinkage * mu
    return shrunk_cov


def oas_cov(emp_cov):
    n_samples, n_features = emp_cov.shape
    alpha = np.mean(emp_cov ** 2)
    mu = np.trace(emp_cov) / n_features
    mu_squared = mu ** 2

    # The factor 1 / p**2 will cancel out since it is in both the numerator and
    # denominator
    num = alpha + mu_squared
    den = (n_samples + 1) * (alpha - mu_squared / n_features)
    shrinkage = 1.0 if den == 0 else min(num / den, 1.0)

    # The shrunk covariance is defined as:
    # (1 - shrinkage) * S + shrinkage * F (cf. Eq. 4 in [1])
    # where S is the empirical covariance and F is the shrinkage target defined as
    # F = trace(S) / n_features * np.identity(n_features) (cf. Eq. 3 in [1])
    shrunk_cov = (1.0 - shrinkage) * emp_cov
    shrunk_cov.flat[:: n_features + 1] += shrinkage * mu
    return shrunk_cov

# if __name__ == '__main__':
# for f in os.listdir('/mnt/f/data/minute_30'):
#     return_entry_info_1(f.split('.')[0])
# pre_averaging_realized_covariance(f.split('.')[0])
# improve_pre_averaging_realized_covariance(f.split('.')[0])
# realized_covariance(f.split('.')[0])
# re_data = []
# flag = 0
# for f in os.listdir('/mnt/f/data/minute_300/FFD_300'):
#     a = pd.read_csv('/mnt/f/data/minute_300/FFD_300/'+f)
#     re_data.append(a)
#     flag +=1
#     if flag==5:
#         data=pd.concat(re_data)
#         re_data=[]
#         flag=0
#         re = realized_covariance_1(data,f.split('.')[0])
# re = return_entry_info_1(f.split('.')[0])
# re_data.append(re)
# re_data = pd.concat(re_data,axis=1)
# re_data = re_data.loc[:, ~re_data.columns.duplicated()]
# re_data.to_csv("/mnt/f/R_test/entry_difference_price.csv",index=False)
