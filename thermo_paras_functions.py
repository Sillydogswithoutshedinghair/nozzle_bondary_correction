# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 20:57:56 2026

@author: admin
"""
import math


# %% 计算空气粘性
def cal_viscosity(Ts):
    """
    已知 Ts，利用 Sutherland's law 求空气的粘性
    Sutherland's law:   https://www.cfd-online.com/Wiki/Sutherland%27s_law
    粘性流体力学, 阎　超　钱翼稷　连祺祥, 北京航空航天大学出版社，2005
    参数:
        Ts    : 静温

    返回:
        viscosity =
    """
    T_ref = 273.15
    viscosity_ref = 1.716e-5
    S = 110.4
    viscosity = viscosity_ref * math.pow(Ts / T_ref, 3.0 / 2.0) * (T_ref + S) / (Ts + S)

    return viscosity


# %% 计算绝热壁面温度 T_aw
def cal_T_aw(T_e, Pr, Ma_e, gama=1.4):
    """
    已知 T_e, Pr, Ma_e, gama

    参数:
        T_e    : 自由流静温，也就是无粘壁面的静温
        Pr    : 普朗特数
        Ma_e  : 自由流 Ma, 也就是无粘壁面的 Ma
        gama  : 比热比（默认 1.4，即空气常用值）
    返回:
        T_aw =
    """

    T_aw = T_e * (1.0 + math.pow(Pr, 1.0 / 3.0) * (gama - 1.0) / 2.0 * Ma_e**2)

    return T_aw


# %% 计算参考温度 T_ref  公式（73）
def cal_T_ref(T_w, T_aw, T_e):
    """
    已知 T_w, Pr, Ma_e, gama

    参数:
        T_w   : 壁面温度
        T_aw  : 壁面绝热温度
        T_e   : 自由流 静温

    返回:
        T_ref =
    """
    B1 = 0.54
    B2 = 0.16
    T_ref = B1 * T_w + B2 * T_aw + (1.0 - B1 - B2) * T_e

    return T_ref


# %% 已知 W, 总温Tt, 求静温 Ts
def cal_Ts(W, Tt, gama=1.4):
    """
    已知 速度因数 W, 总温Tt，比热比gama 计算 静温

    参数:
        W     : 速度因数
        Tt    : 总温
        gama  : 比热比（默认 1.4，即空气常用值）

    返回:
        Ts = 静温，见《气体动力学基础》王新月，西北工业大学出版社 P123
    """
    Ts = Tt * (1.0 - (gama - 1.0) / (gama + 1.0) * W**2)

    return Ts


# %% 已知 W, 总压Pt, 求静压 Ps
def cal_Ps(W, Pt, gama=1.4):
    """
    已知 速度因数 W, 总压 Pt，比热比gama 计算 静压

    参数:
        W     : 速度因数
        Pt    : 总压
        gama  : 比热比（默认 1.4，即空气常用值）

    返回:
        Ps = ，静压 见《气体动力学基础》王新月，西北工业大学出版社 P123
    """
    Ps = Pt * math.pow((1.0 - (gama - 1.0) / (gama + 1.0) * W**2), gama / (gama - 1.0))

    return Ps
