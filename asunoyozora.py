from typing import Any, Callable, Final, NewType, Union

import numpy as np
import numpy.typing as npt
from numpy import sqrt

Meter = NewType('Meter', float)
GPMeter = NewType('GPMeter', float)
Kelvin = NewType('Kelvin', float)
Pascal = NewType('Pascal', float)
Kilogram = NewType('Kilogram', float)
Second = NewType('Second', float)
SqueareMeter = NewType('SqueareMeter', float)

Numeric = Union[float, int, npt.NDArray[np.float64]]

ln: Callable[[Numeric], Numeric] = np.log
cosh: Callable[[Any], Any] = np.cosh

# std = STDATM()


# T_b:

# for i in H_i:
#     T_b.append(std.T(std.H2Z(GPMeter(i * 1000))))


# P_b:

# index: int = 0
# for i in L_M_b:
#     if index == 0:
#         P_b.append(Pascal(round(float(std.P_33b(H_i[index])), 12)))
#     else:
#         P_temp = P_b[-1]
#         if abs(i) > 1e-9:
#             P_b.append(Pascal(round(float(std.P_33a(H_i[index], P_temp)), 12)))
#         else:
#             P_b.append(Pascal(round(float(std.P_33b(H_i[index], P_temp)), 12)))
#     index += 1


R: Final[float] = 8.314462618 * (10**3)
g_0: Final[float] = 9.80665
M_0: Final[float] = 28.9644
r_0: Final[Meter] = Meter(6356.766 * (10**3))
T_0: Final[Kelvin] = Kelvin(218.5)
P_0: Final[Pascal] = Pascal(101325)
H_i: Final[list[GPMeter]] = [GPMeter(0 * 1000), GPMeter(11 * 1000), GPMeter(20 * 1000), GPMeter(32 * 1000), GPMeter(47 * 1000), GPMeter(51 * 1000), GPMeter(71 * 1000), GPMeter(84.852 * 1000)]
P_b: Final[list[Pascal]] = [Pascal(101325.0), Pascal(17881.924167776844), Pascal(4451.782721266618), Pascal(835.676386546647), Pascal(125.823085025137), Pascal(75.131157658166), Pascal(2.218089806735), Pascal(0.010142033211)]
T_M_b: Final[list[Kelvin]] = [Kelvin(288.15), Kelvin(216.65), Kelvin(216.65), Kelvin(228.65), Kelvin(270.65), Kelvin(270.65), Kelvin(214.65), Kelvin(186.95)]
L_M_b: Final[list[float]] = [-6.5 / 1000, 0.0 / 1000, 1.0 / 1000, 2.8 / 1000, 0.0 / 1000, -2.8 / 1000, -2.0 / 1000, 0.0 / 1000]


class STDATM:

    def __init__(self, H_t: Meter, H_h: Meter, m: Kilogram):
        """init

        Parameters
        ----------
        H_t : Meter
            ヒトの厚み(背面から前面)
        H_h : Meter
            ヒトの身長
        m : Kilogram
            ヒトの質量
        """
        self.H_t: Meter = H_t
        self.H_h: Meter = H_h
        self.m: Kilogram = m

    def analytical_Z(self, t: Second, base_Z: Meter) -> Meter:
        """解析解的に高度を求める方法rhoの値がbaseZで決定される

        Parameters
        ----------
        t : Second
            落下時間
        base_Z : Meter
            計算時の高度

        Returns
        -------
        Meter
            初期高度
        """
        local_k = self.k(base_Z)
        analytical_Z_val = sqrt((local_k * g_0) / self.m) * t
        # coshxについて考える時、numpyではx > 710.47586007394392の場合np.infになるため、例外処理をする
        if analytical_Z_val < 710.47586007394392:
            analytical_result: Meter = Meter((self.m / local_k) * float(ln(cosh(analytical_Z_val))))
        else:
            # cosh(x) ≈ e^x / 2 ⇒ log(cosh(x)) ≈ x - log(2)
            analytical_result: Meter = Meter((self.m / local_k) * float(analytical_Z_val - ln(2)))
        return analytical_result

        # Meter((50 / local_k) * float(ln(cosh(sqrt((local_k * g_0) / 50) * t))))

    def a(self, Z: Meter, v: float) -> float:
        return float(g_0 - ((self.k(Z) * (v ** 2)) / self.m))

    def Cd(self) -> float:
        # 難しいから定数 1である妥当性はない
        return float(1)

    def k(self, Z: Meter) -> float:
        """空気抵抗係数

        Parameters
        ----------
        Z : Meter
            高度

        Returns
        -------
        float
            係数
        """
        k_result: float = (1 / 2) * self.A(self.H_t, self.H_h) * self.Cd() * self.rho(Z)
        return k_result

    def A(self, H_t: Meter, H_h: Meter) -> SqueareMeter:
        """ヒトの平面図の面積

        Parameters
        ----------
        H_t : Meter
            ヒトの厚み(背面から前面)
        H_h : Meter
            ヒトの身長

        Returns
        -------
        SqueareMeter
            人の平面図を長方形に近似した面積
        """
        # H_width_rate (0.22 ~ 0.27)
        H_width_rate: float = 0.24
        # H_widthは肩幅
        H_width: Meter = Meter(H_h * H_width_rate)
        return SqueareMeter(H_t * H_width)

    def rho(self, Z: Meter) -> float:
        """空気密度\\rhoを求めます

        Parameters
        ----------
        Z : Meter
            _description_

        Returns
        -------
        float
            _description_
        """
        rho_result: float = (self.P(Z) * M_0) / (R * self.T(Z))
        return rho_result

    def T(self, Z: Meter) -> Kelvin:
        """Templeture

        Parameters
        ----------
        Z : Meter
            高度

        Returns
        -------
        Kelvin
            Templeture
        """
        h_layers_m = H_i    # ジオポテンシャル高度の境界リスト [km]
        beta_layers_k_per_m = L_M_b  # 各層の温度勾配リスト [K/km]
        t_ground_k = 288.15  # 地表の基準温度 [K]

        # 1. 幾何高度(Z)をジオポテンシャル高度(H)に変換 [m]
        H_m: GPMeter = self.Z2H(Z)

        # 2. メートル単位での計算に備える

        # 3. 地表温度から計算を開始
        temp_k = t_ground_k

        # 4. 各層を通過する際の温度変化を積算
        for i in range(len(beta_layers_k_per_m)):
            h_bottom = h_layers_m[i]
            h_top = h_layers_m[i + 1]
            beta = beta_layers_k_per_m[i]

            if H_m < h_top:

                temp_k += beta * (H_m - h_bottom)
                return Kelvin(temp_k)
            else:

                temp_k += beta * (h_top - h_bottom)

        return Kelvin(temp_k)

    def get_b(self, H: GPMeter) -> int:
        """どの層bに所属しているかを判定します

        Parameters
        ----------
        Z : Meter
            高度

        Returns
        -------
        int
            bの値
        """
        if H == 0:
            if H_i[0] == 0:
                return 0
        for i in range(len(H_i) - 1):
            H_base = H_i[i]
            H_top = H_i[i + 1]
            if H_base <= H < H_top:
                return i
        if H == H_i[-1] and len(H_i) > 1:
            return len(H_i) - 2
        return -1

    def P(self, Z: Meter) -> Pascal:
        H = self.Z2H(Z)
        b = self.get_b(H)

        return self.P_33b(H, P_b[b]) if L_M_b[b] == 0 else self.P_33a(H, P_b[b])

    def P_33a(self, H: GPMeter, P_b: Pascal = Pascal(0)) -> Pascal:
        """L \\not = 0

        Parameters
        ----------
        H : GPMeter
            _description_
        P_b : Pascal, optional
            _description_, by default Pascal(0)

        Returns
        -------
        Pascal
            _description_
        """
        b = self.get_b(H)
        # print(f"高度H={H}, b={b}")
        P33a_value = (g_0 * M_0) / (R * L_M_b[b])
        result = P_b * ((T_M_b[b]) / (T_M_b[b] + L_M_b[b] * (H - H_i[b - 1]))) ** P33a_value
        return result

    def P_33b(self, H: GPMeter, P_b: Pascal = Pascal(0)) -> Pascal:
        """L = 0

        Parameters
        ----------
        H : GPMeter
            _description_
        P_b : Pascal, optional
            _description_, by default Pascal(0)

        Returns
        -------
        Pascal
            _description_
        """
        b = self.get_b(H)
        # print(f"高度H={H}, b={b}")
        P33b_value = (g_0 * M_0) / (R * T_M_b[b])
        result = P_b * np.exp(-1 * P33b_value * (H - H_i[b - 1]))
        return result

    def Z2H(self, Z: Meter) -> GPMeter:
        """Meter to GPMeter

        Parameters
        ----------
        Z : Meter

        Returns
        -------
        GPMeter
        """
        return GPMeter((r_0 * Z) / (r_0 + Z))

    def H2Z(self, H: GPMeter) -> Meter:
        """GPMeter to Meter

        Parameters
        ----------
        H : GPMeter

        Returns
        -------
        Meter
        """
        return Meter((H * r_0) / (r_0 - H))


def analytical_solution(t: Second) -> tuple[Meter, float]:
    """
    f_simpleに対応する解析解（理論値）を計算する。

    引数:
        t (float or np.array): 時刻

    戻り値:
        tuple: (Z_analytical, v_analytical)
               時刻tにおける理論上の落下距離と速度
    """
    # Z(t) = 0.5 * g * t^2
    # v(t) = g * t
    Z_analytical: Meter = Meter(0.5 * g_0 * t**2)
    v_analytical: float = g_0 * t
    return Z_analytical, v_analytical


def f_simple(t: Second, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    RK4の動作テスト用。超単純な自由落下モデル。
    空気抵抗なし、重力は一定。鉛直下向きが正。

    引数:
        t (float): 現在の時刻。この単純モデルでは使わない。
        y (np.array): 状態ベクトル [Z, v]。Zは落下距離、vは速度。

    戻り値:
        np.array: yの時間変化率 [dZ/dt, dv/dt]。
    """

    # 状態ベクトルを分かりやすい変数に分解
    Z, v = y  # type: ignore

    # y の時間変化率 dy/dt を計算
    # dy/dt = [dZ/dt, dv/dt]

    # dZ/dt = v (距離の変化率は、現在の速度)
    dZ_dt = v

    # dv/dt = g (速度の変化率は、一定の重力加速度)
    dv_dt = g_0

    # 結果をNumPy配列で返す
    return np.array([dZ_dt, dv_dt], dtype=np.float64)


def f_real(t: Second, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    Z, v = y
    dZ_dt = v
    dv_dt = std.a(Z, v)
    return np.array([dZ_dt, dv_dt], dtype=np.float64)


def rk4(f: Callable[[Second, npt.NDArray[np.float64]], npt.NDArray[np.float64]], y0: npt.NDArray[np.float64], t0: Second, t1: Second, dt: Second) -> npt.NDArray[np.float64]:
    """calc. for RK4

    Parameters
    ----------
    f : Callable[[Second, : npt.NDArray[np.float64]], : npt.NDArray[np.float64]]
        calc function
    y0 : npt.NDArray[np.float64]
        初期vector
    t0 : Second
        start sec
    t1 : Second
        end sec
    dt : Second
        small step

    Returns
    -------
    ndarray
        演算結果[Z, v]
    """
    steps = int((t1 - t0) / dt)
    results: npt.NDArray[np.float64] = np.zeros((steps + 1, len(y0)), dtype=np.float64)
    results[0] = y0
    time_points = np.linspace(t0, t1, steps + 1)
    y_0: npt.NDArray[np.float64] = y0.copy()
    # print(f"{t0}~{t1}まで1ステップを{dt}とし、演算を開始します")
    for i in range(steps):
        t = time_points[i]
        k1: npt.NDArray[np.float64] = f(t, y_0)
        k2: npt.NDArray[np.float64] = f(t + dt / 2, np.array(y_0 + k1 * (dt / 2), dtype=np.float64))
        k3: npt.NDArray[np.float64] = f(t + dt / 2, np.array(y_0 + k2 * (dt / 2), dtype=np.float64))
        k4: npt.NDArray[np.float64] = f(t + dt, np.array(y_0 + k3 * dt, dtype=np.float64))
        y_0 += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        results[i + 1] = y_0
    # print(f"y:{y_0}")
    # print(f"t:{time_points[-1]}")
    return y_0


def test_rk4():
    step: Final[Second] = Second(0.01)
    end_time: Final[Second] = Second(177)
    start_time: Final[Second] = Second(0)

    initial_y: npt.NDArray[np.float64] = np.array([0.0, 0.0], dtype=np.float64)

    rk4_result: npt.NDArray[np.float64] = rk4(f=f_simple, y0=initial_y, t0=start_time, t1=end_time, dt=step)
    true_result: tuple[Meter, float] = analytical_solution(end_time)
    true_result_array: npt.NDArray[np.float64] = np.array(true_result)

    print(f"y: {true_result}")

    assert np.allclose(rk4_result, true_result_array)

    print("\nAssertion successful! RK4 implementation is correct.")


def main(start_time: Second, endtime: Second, step: Second, tol: Meter, std: STDATM):
    analytical_Zlist: list[Meter] = [Meter(0)]
    for i in range(10):
        analytical_Zlist.append(std.analytical_Z(endtime, analytical_Zlist[i]))

    # 以下二分探索
    Z_base: Meter = analytical_Zlist[-1]
    low = Z_base / 2
    high = Z_base * 2
    while high - low > tol:
        mid = (low + high) / 2
        rk4_result = rk4(f_real, np.array([mid, 0.0], dtype=np.float64), start_time, endtime, step)
        if rk4_result[0] > 0:
            high = mid
        else:
            low = mid


if __name__ == "__main__":
    print("start")

    # RK4用の変数の宣言
    start_time: Second = Second(0)
    endtime: Second = Second(177)
    step: Second = Second(0.01)
    # 許容できる誤差
    tol: Meter = Meter(1e-3)

    # STDATMのインスタンスを作る
    body_weight: Kilogram = Kilogram(50)
    Human_height: Meter = Meter(1.7)
    Human_thickness: Meter = Meter(0.2)
    std = STDATM(H_t=Human_thickness, H_h=Human_height, m=body_weight)

    main(start_time, endtime, step, tol, std)
