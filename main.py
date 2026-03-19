import pickle
import numpy as np
from pathlib import Path
import thermo_paras_functions
import math


# 从pkl文件读取后，会使用到的参数
contour_wall: np.ndarray = None  # 无粘外形参数
Pt: float = None  # 总压 # unit Pa
Tt: float = None  # 总温 # unit K
Rg: float = None  # 气体常数
R: float = None  # ratio of throat radius of curvature to throat radius
r1: float = None  # value of r when Ma = 1; unit m
r_star: float = None  # radius of geometric throat; unit m
R1: float = None  # r1/r_star

cw_dict: dict = None  # 无粘外形参数，字典
N: int = None  # 总离散点数
Tw: float = 300.0  # K
# alpha 边界层温度型因子，Eq.62。对结果影响较大。不建议为0，对于喉部的高静温而言，会导致H为负值
alpha: float = 1.0
gama: float = 1.4  # 比热比
Pr: float = 0.72  # 普朗特数
criterion_Re_theta = 0.001  # 迭代计算Re_theta时，相邻两次Re_theta相对误差的准则

delta_a_throat = 3e-4  # 指定喉部(n=1)的轴对称边界层位移厚度; unit m


def load_variables():
    # 加载pkl文件的参数
    with open("variables.pkl", "rb") as f:
        variables = pickle.load(f)

    globals().update(variables)

    # 注意第11个点开始才是H点，即n=1点，此处先把前十个点去除
    n_start = len(contour_TH) - 1
    contour_wall_H = contour_wall[n_start:, :]

    # contour_wall 每列参数名
    keys = ["x", "y", "r", "W", "Ma", "phi", "T_s", "c", "V", "Vx", "V_y", "masssflow"]

    global cw_dict, N, r_star
    cw_dict = dict(zip(keys, contour_wall_H.T))
    N = len(cw_dict["x"])
    r_star = r1 / R1

    # 补充已知参数
    miu_e = [thermo_paras_functions.cal_viscosity(ts) for ts in cw_dict["T_s"]]
    miu_w = np.ones((N,)) * thermo_paras_functions.cal_viscosity(Tw)
    T_aw = [
        thermo_paras_functions.cal_T_aw(te, Pr, ma_e)
        for te, ma_e in zip(cw_dict["T_s"], cw_dict["Ma"])
    ]
    T_ref = [
        thermo_paras_functions.cal_T_ref(Tw, taw, te)
        for taw, te in zip(T_aw, cw_dict["T_s"])
    ]
    p_s = [thermo_paras_functions.cal_Ps(W, Pt) for W in cw_dict["W"]]  # 静压

    cw_dict["miu_e"] = np.asarray(miu_e)
    cw_dict["miu_w"] = miu_w
    cw_dict["T_aw"] = np.asarray(T_aw)
    cw_dict["T_ref"] = np.asarray(T_ref)
    cw_dict["p_s"] = np.asarray(p_s)
    cw_dict["rho_s"] = cw_dict["p_s"] / Rg / cw_dict["T_s"]  # p = rho * R * T

    # 待求参数
    cw_dict["Re_theta"] = np.zeros((N,))
    cw_dict["Re_theta_i"] = np.zeros((N,))
    cw_dict["theta"] = np.zeros((N,))
    cw_dict["theta_deri"] = np.zeros((N,))
    cw_dict["H"] = np.zeros((N,))
    cw_dict["H_i"] = np.zeros((N,))
    cw_dict["C_f"] = np.zeros((N,))
    cw_dict["C_fi"] = np.zeros((N,))
    cw_dict["P"] = np.zeros((N,))
    cw_dict["Q"] = np.zeros((N,))

    cw_dict["delta_star"] = np.zeros((N,))
    cw_dict["delta_a_star"] = np.zeros((N,))
    cw_dict["boundary_correction"] = np.zeros((N,))

    cw_dict["iter"] = np.zeros((N,))  # 记录各点迭代次数


def cal_H(n):
    """
    Result:
    ---------------
    形状因子，Eq.68
    """
    H = (
        -1
        + Tw * (cw_dict["H_i"][n] + 1) / cw_dict["T_s"][n]
        + alpha * (cw_dict["T_aw"][n] - Tw) / cw_dict["T_s"][n]
    )
    return H


def cal_Hi(n):
    """
    Result:
    --------------
    不可压缩形状因子 Eq.72
    """
    H_i = (1 - 7 * (cw_dict["C_fi"][n] / 2) ** 0.5) ** (-1)
    return H_i


def cal_Cf(n):
    """
    Result:
    -------------
    摩擦系数，Eq.70
    """
    C_f = cw_dict["T_s"][n] / cw_dict["T_ref"][n] * cw_dict["C_fi"][n]
    return C_f


def cal_Cfi(n):
    """
    Result:
    ---------------
    不可压摩擦因子，Eq.70
    """
    C_fi = 0.0773 / (
        (math.log10(cw_dict["Re_theta_i"][n].item()) + 4.563)
        * (math.log10(cw_dict["Re_theta_i"][n].item()) - 0.546)
    )
    return C_fi


def cal_Re_theta_i(n):
    """
    Result:
    ---------------
    计算基于动量厚度的不可压雷诺数，Eq.71
    """
    Re_theta_i = (
        Tw
        * cw_dict["miu_e"][n]
        * cw_dict["Re_theta"][n]
        / cw_dict["T_ref"][n]
        / cw_dict["miu_w"][n]
    )
    return Re_theta_i


def cal_Re_theta(n):
    """
    Result:
    --------------
    计算基于动量厚度的雷诺数，Re = rho_e * u_e * theta / miu_e
    """
    Re_theta = (
        cw_dict["rho_s"][n]
        * cw_dict["V"][n]
        * cw_dict["theta"][n]
        / cw_dict["miu_e"][n]
    )
    return Re_theta


def update_H_Cf(n, iter):
    """
    1. iter = 0 初始化Re_theta, 依次初始化Re_theta_i, C_fi, Cf, H_i, H
    2. iter > 0 根据最新计算的theta，更新当前位点n的Re_theta_i, C_fi, Cf, H_i, H
    params:
    ----------------
        n：当前位点的index，n=(0,1,2,3...N)
        iter: 迭代次数

    result:
    ----------------
    更新cw_dict的{Re_theta, Re_theta_i, C_fi, C_f, Hi, H}
    """
    global cw_dict

    # 点n的第一次迭代
    if iter == 0:
        # 用上一个station的值来赋初值
        cw_dict["Re_theta"][n] = (
            cw_dict["Re_theta"][n - 1] if n > 0 else 1e4
        )  # 1E4初值对结果没有显著影响
        if n == 2:
            cw_dict["Re_theta"][n] = cw_dict["Re_theta"][0]
            # 不用n=1是因为此时还未求解2

        cw_dict["Re_theta_i"][n] = cal_Re_theta_i(n)

        cw_dict["C_fi"][n] = cal_Cfi(n)
        cw_dict["C_f"][n] = cal_Cf(n)

        cw_dict["H_i"][n] = cal_Hi(n)
        cw_dict["H"][n] = cal_H(n)
        # # =============
        # # 应对alpha为0时，可能出现的H为负数情况。
        # if cw_dict["H"][n] <= 0 and n == 0:
        #     cw_dict["H"][n] = 0.3
        # # ----------------
        return iter

    if iter > 0:
        # 基于iter-1计算的theta，计算Re_theta
        temp_Re_theta = cal_Re_theta(n)

        if (
            abs(temp_Re_theta - cw_dict["Re_theta"][n]) / cw_dict["Re_theta"][n]
            <= criterion_Re_theta
        ):
            cw_dict["Re_theta"][n] = temp_Re_theta
            cw_dict["iter"][n] = iter
            iter = -1
            return iter
        else:
            cw_dict["Re_theta"][n] = temp_Re_theta
            cw_dict["Re_theta_i"][n] = cal_Re_theta_i(n)
            cw_dict["C_fi"][n] = cal_Cfi(n)
            cw_dict["C_f"][n] = cal_Cf(n)
            cw_dict["H_i"][n] = cal_Hi(n)
            cw_dict["H"][n] = cal_H(n)
            # # ==============
            # # 应对alpha为0时，可能出现的H为负数情况。
            # if cw_dict["H"][n] <= 0 and n == 0:
            #     cw_dict["H"][n] = 0.3
            # # ============

            return iter


def n0(delta_a_star):
    """
    计算n=0点的theta，且已知theta' = 0

    params:
    ------------------
    delta_a_star: 指定n=0处的位移厚度
    """
    global cw_dict

    n = 0

    # 根据delta_a_star和Eq.61，反解delta_star。推导略
    delta_star = (
        delta_a_star**2
        + 2
        * delta_a_star
        * cw_dict["y"][0]
        / math.cos(math.radians(cw_dict["phi"][0].item()))
    ) / (
        2 * delta_a_star
        + 2 * cw_dict["y"][0] / math.cos(math.radians(cw_dict["phi"][0].item()))
    )  # 反解Eq.61

    W_0 = (
        1
        + 1 / (4 * R)
        - (14 * gama + 15) / (288 * R**2)
        + (2364 * gama**2 + 4149 * gama + 2241) / (82944 * R**3)
    )  # Eq. 57

    beta = (2 / (gama + 1) * R) ** (0.5)
    dW_dx_0 = (beta / r_star) * (
        1 + 3 / (8 * R) - (64 * gama**2 + 117 * gama + 54) / (1154 * R**2)
    )  # Eq. 58

    for iter in range(10000):
        iter = update_H_Cf(n, iter)  # 更新{Re_theta, Re_theta_i, C_fi, C_f, Hi, H}
        if iter == -1:
            break
        cw_dict["theta"][0] = delta_star / cw_dict["H"][0]

    cw_dict["P"][0] = (2 + cw_dict["H"][0] - cw_dict["Ma"][0] ** 2) * dW_dx_0 / W_0
    cw_dict["Q"][0] = cw_dict["theta"][0] * cw_dict["P"][0]
    cw_dict["theta_deri"][0] = 0  # 人工指定

    # print(f"n={n}, theta = {cw_dict["theta"][n]}")


def n1n2():
    """
    将theta_n1和theta'_n1用theta_n2和theta_n2替换，从而求解Eq.56，得到theta_n2
    再计算得到theta_n1
    """
    # %%先求解theta_n2，在文章中是点3
    global cw_dict

    n = 2

    for iter in range(10000):
        iter = update_H_Cf(n, iter)  # 更新{Re_theta, Re_theta_i, C_fi, C_f, Hi, H}
        if iter == -1:
            break

        s = cw_dict["x"][n - 1] - cw_dict["x"][n - 2]
        t = cw_dict["x"][n] - cw_dict["x"][n - 1]

        P_n1 = 2 - cw_dict["Ma"][n] ** 2 + cw_dict["H"][n]
        P_n2 = cw_dict["Ma"][n] * (1 + (gama - 1) / 2 * cw_dict["Ma"][n] ** 2)
        P_n3 = (
            t**2 * cw_dict["Ma"][n - 2]
            - (s + t) ** 2 * cw_dict["Ma"][n - 1]
            + (s + 2 * t) * s * cw_dict["Ma"][n]
        ) / (
            s * (s + t) * t
        )  # dM/dx，向后差分格式
        P_n4 = (
            1
            / cw_dict["y"][n]
            * (
                t**2 * cw_dict["y"][n - 2]
                - (s + t) ** 2 * cw_dict["y"][n - 1]
                + (s + 2 * t) * s * cw_dict["y"][n]
            )
            / (s * (s + t) * t)
        )  # dy/dx，向后差分格式
        cw_dict["P"][n] = P_n1 / P_n2 * P_n3 + P_n4

        cw_dict["Q"][n] = (
            cw_dict["C_f"][n] / 2 / math.cos(math.radians(cw_dict["phi"][n].item()))
        )

        # Eq. 52-54
        Gn_2 = (2 * s - t) * (s + t) / 6 / s
        Gn_1 = (s + t) ** 3 / 6 / s / t
        Gn = (2 * t - s) * (s + t) / 6 / t

        # 将Eq.56中的theta_n1替换为theta_n2得到的公式，求解theta_2
        cw_dict["theta"][n] = (
            cw_dict["theta"][n - 2]
            + Gn_2 * cw_dict["theta_deri"][n - 2]
            + Gn * cw_dict["Q"][n]
            + s / (s + t) * Gn_1 * cw_dict["Q"][n]
        ) / (1 + Gn * cw_dict["P"][n] + s / (s + t) * Gn_1 * cw_dict["P"][n])

        cw_dict["theta_deri"][n] = (
            cw_dict["Q"][n] - cw_dict["P"][n] * cw_dict["theta"][n]
        )

    # %% 然后计算n=1点
    n = 1
    cw_dict["theta"][n] = (
        s**2 / (s + t) ** 2 * (cw_dict["theta"][2] - cw_dict["theta"][0])
        + cw_dict["theta"][0]
    )
    cw_dict["theta_deri"][n] = s * cw_dict["theta_deri"][2] / (s + t)
    cw_dict["Re_theta"][n] = cal_Re_theta(n)
    cw_dict["Re_theta_i"][n] = cal_Re_theta_i(n)
    cw_dict["C_fi"][n] = cal_Cfi(n)
    cw_dict["C_f"][n] = cal_Cf(n)
    cw_dict["H_i"][n] = cal_Hi(n)
    cw_dict["H"][n] = cal_H(n)

    # print(f"n={1}, theta = {cw_dict["theta"][1]}")
    # print(f"n={2}, theta = {cw_dict["theta"][2]}")


def solve_momentum_eq():
    """
    对于n>=2的空间离散点，求解动量方程。求解theta

    result
    ----------------
    更新cw_dict中，点n的{theta, theta_deri, P, Q}参数
    """
    global cw_dict

    for n in range(3, N):
        s = cw_dict["x"][n - 1] - cw_dict["x"][n - 2]
        t = cw_dict["x"][n] - cw_dict["x"][n - 1]

        Gn_2 = (2 * s - t) * (s + t) / 6 / s
        Gn_1 = (s + t) ** 3 / 6 / s / t
        Gn = (2 * t - s) * (s + t) / 6 / t

        P_n2 = cw_dict["Ma"][n] * (1 + (gama - 1) / 2 * cw_dict["Ma"][n] ** 2)
        # dM/dx，向后差分格式
        P_n3 = (
            t**2 * cw_dict["Ma"][n - 2]
            - (s + t) ** 2 * cw_dict["Ma"][n - 1]
            + (s + 2 * t) * s * cw_dict["Ma"][n]
        ) / (s * (s + t) * t)
        # 1/y*dy/dx，向后差分格式
        P_n4 = (
            1
            / cw_dict["y"][n]
            * (
                t**2 * cw_dict["y"][n - 2]
                - (s + t) ** 2 * cw_dict["y"][n - 1]
                + (s + 2 * t) * s * cw_dict["y"][n]
            )
            / (s * (s + t) * t)
        )

        for iter in range(10000):
            iter = update_H_Cf(n, iter)  # 更新{Re_theta, Re_theta_i, C_fi, C_f, Hi, H}
            if iter == -1:
                # print(f"n={n}, theta = {cw_dict["theta"][n]}")
                break

            P_n1 = 2 - cw_dict["Ma"][n] ** 2 + cw_dict["H"][n]

            cw_dict["P"][n] = P_n1 / P_n2 * P_n3 + P_n4

            cw_dict["Q"][n] = (
                cw_dict["C_f"][n] / 2 / math.cos(math.radians(cw_dict["phi"][n].item()))
            )

            cw_dict["theta"][n] = (
                cw_dict["theta"][n - 2]
                + Gn_2 * cw_dict["theta_deri"][n - 2]
                + Gn_1 * cw_dict["theta_deri"][n - 1]
                + Gn * cw_dict["Q"][n]
                # ) / (1 + cw_dict["Q"][n] * cw_dict["P"][n])
            ) / (1 + Gn * cw_dict["P"][n])

            cw_dict["theta_deri"][n] = (
                cw_dict["Q"][n] - cw_dict["P"][n] * cw_dict["theta"][n]
            )


def cal_boundary_correction():
    """
    基于求解的动量厚度theta，修正边界层
    """
    global cw_dict

    cw_dict["delta_star"] = cw_dict["H"] * cw_dict["theta"]

    cw_dict["delta_a_star"] = (
        cw_dict["delta_star"]
        + (
            cw_dict["delta_star"] ** 2
            + cw_dict["y"] ** 2 / (np.cos(np.deg2rad(cw_dict["phi"]))) ** 2
        )
        ** (1 / 2)
        - cw_dict["y"] / np.cos(np.deg2rad(cw_dict["phi"]))
    )

    cw_dict["boundary_correction"] = cw_dict["delta_a_star"] / np.cos(
        np.deg2rad(cw_dict["phi"])
    )


def over_plot():
    from matplotlib import pyplot as plt

    plt.figure()
    plt.plot(cw_dict["x"], cw_dict["y"])
    plt.plot(
        cw_dict["x"],
        cw_dict["y"] - cw_dict["boundary_correction"],
        linestyle="--",
        color="red",
    )
    plt.ylim(0, 0.6)
    plt.axis("scaled")
    plt.text(
        x=cw_dict["x"][0],
        y=cw_dict["y"][-1] * 2,
        s=f"The boundary correction at the exit is {cw_dict["boundary_correction"][-1]:.4f} m",
    )
    plt.show()


def main():
    # 加载无粘外形参数
    load_variables()

    n0(delta_a_star=delta_a_throat)  # unit m
    n1n2()

    solve_momentum_eq()

    cal_boundary_correction()

    over_plot()


if __name__ == "__main__":
    main()
