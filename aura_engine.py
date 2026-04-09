import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline
import json
import traceback


class WebProfileManager:
    def __init__(self):
        self.profiles = {"Public Mode": {"c_R": 3.0, "c_G": 2.0, "baseline_bpm": 75.0, "trained": False}}
        self.current_user = "Public Mode"
        # 新增：用於穩定心跳輸出的中位數緩衝區
        self.bpm_history = []

    def update_learning(self, bpm_truth, r_mean, g_mean):
        p = self.profiles.get(self.current_user)
        if not p or not p.get("trained", False): return

        # 限制學習極限：強制鎖定在人體膚色的物理極限 (2.5 ~ 3.5 之間)
        opt_R = r_mean / g_mean if g_mean > 0 else 3.0
        opt_R = max(2.5, min(3.5, opt_R))

        # 降低學習速率，避免一次錯誤光線污染數據
        p["c_R"] = (p["c_R"] * 0.99) + (opt_R * 0.01)
        p["baseline_bpm"] = (p["baseline_bpm"] * 0.95) + (bpm_truth * 0.05)


engine = WebProfileManager()


def sync_profiles(json_str):
    try:
        data = json.loads(json_str)
        engine.profiles.update(data)
    except:
        pass


def export_profiles():
    return json.dumps(engine.profiles)


def process_data_from_js(rgb_data, fps, polar_bpm, profile_name, is_training, is_training_active):
    try:
        # 切換使用者時，自動清空歷史穩定緩衝區，避免幽靈數據殘留
        if engine.current_user != profile_name:
            engine.bpm_history = []
            engine.current_user = profile_name

        if hasattr(rgb_data, 'to_py'):
            rgb_data = rgb_data.to_py()
        buffer = np.asarray(rgb_data, dtype=np.float64).reshape(-1, 3)

        if len(buffer) < 150:
            return {"bpm": 0, "rmssd": 0, "sdnn": 0, "error": "緩衝資料不足", "peaks": 0}

        R, G, B = buffer[:, 0], buffer[:, 1], buffer[:, 2]

        if np.mean(G) < 1:
            return {"bpm": 0, "rmssd": 0, "sdnn": 0, "error": "影像全黑 (請檢查iOS設定)", "peaks": 0}

        p = engine.profiles.get(profile_name, engine.profiles["Public Mode"])

        # CHROM 演算法
        X = p["c_R"] * R - p["c_G"] * G
        Y = 1.5 * R + G - 1.5 * B

        std_Y = np.std(Y)
        if std_Y == 0: std_Y = 1e-6
        alpha = np.std(X) / std_Y

        # 1. 訊號去趨勢 (Detrend)：移除手機晃動產生的基線漂移
        bvp = signal.detrend(X - alpha * Y)

        # 2. 帶通濾波 (Bandpass)：收窄至 45~150 BPM
        nyq = fps / 2.0
        low_cut = 0.75 / nyq  # 45 BPM
        high_cut = 2.5 / nyq  # 150 BPM
        peak_dist_sec = 0.4  # 預設波峰最小距離 (對應最大 150 BPM)

        if is_training and polar_bpm > 30:
            target_hz = polar_bpm / 60.0
            low_cut = max(0.5, target_hz - 0.3) / nyq
            high_cut = min(3.0, target_hz + 0.3) / nyq
            peak_dist_sec = 1.0 / (target_hz + 0.5)
            if is_training_active:
                engine.update_learning(polar_bpm, np.mean(R), np.mean(G))

        # 升級為 4 階 Butterworth 濾波器，邊緣滾降更乾淨
        b, a = signal.butter(4, [low_cut, high_cut], btype='band')
        f_bvp = signal.filtfilt(b, a, bvp)

        # 3. 亞像素波峰分析 (Cubic Spline Interpolation)
        # 將 30fps 的波形上採樣 4 倍至 120fps，大幅提升 HRV 時間解析度
        interp_factor = 4
        new_fps = fps * interp_factor
        t = np.arange(len(f_bvp))
        t_new = np.linspace(0, len(f_bvp) - 1, len(f_bvp) * interp_factor)
        cs = CubicSpline(t, f_bvp)
        f_bvp_interp = cs(t_new)

        # 在高解析度波形上尋找波峰
        peak_dist_interp = int(new_fps * peak_dist_sec)
        prominence_val = max(np.std(f_bvp_interp) * 0.15, 1e-5)
        peaks, _ = signal.find_peaks(f_bvp_interp, distance=peak_dist_interp, prominence=prominence_val)

        bpm, rmssd, sdnn = 0, 0, 0
        if len(peaks) >= 4:
            # 計算 RR 間期 (ms)
            rr = np.diff(peaks) * (1000.0 / new_fps)

            # 4. 嚴格生理過濾器：只允許 400ms(150bpm) ~ 1500ms(40bpm) 的有效跳動
            valid_rr = rr[(rr >= 400) & (rr <= 1500)]

            if len(valid_rr) >= 2:
                # 剔除與中位數落差超過 20% 的異常假波峰
                median_rr = np.median(valid_rr)
                valid_rr = valid_rr[np.abs(valid_rr - median_rr) < (median_rr * 0.2)]

                if len(valid_rr) >= 2:
                    raw_bpm = 60000.0 / np.median(valid_rr)

                    # 5. 輸出穩定化：取最近 5 次的 BPM 中位數，防止數值暴跳
                    engine.bpm_history.append(raw_bpm)
                    if len(engine.bpm_history) > 5:
                        engine.bpm_history.pop(0)
                    bpm = np.median(engine.bpm_history)

                    if len(valid_rr) > 2:
                        diff_rr = np.abs(np.diff(valid_rr))
                        # RMSSD 防爆：過濾大於 150ms 的不合理相鄰跳變
                        clean_diffs = diff_rr[diff_rr < 150]
                        if len(clean_diffs) > 0:
                            rmssd = np.sqrt(np.mean(clean_diffs ** 2))
                        else:
                            rmssd = 0

                    sdnn = np.std(valid_rr)

        return {
            "bpm": round(bpm, 1),
            "rmssd": round(rmssd, 1),
            "sdnn": round(sdnn, 1),
            "error": "",
            "peaks": len(peaks)
        }
    except Exception as e:
        return {"bpm": 0, "rmssd": 0, "sdnn": 0, "error": str(traceback.format_exc()), "peaks": 0}