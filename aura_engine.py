import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline
import json
import traceback


class WebProfileManager:
    def __init__(self):
        self.profiles = {"Public Mode": {"c_R": 3.0, "c_G": 2.0, "baseline_bpm": 75.0, "trained": False}}
        self.current_user = "Public Mode"
        # 用於穩定心跳輸出的中位數緩衝區
        self.bpm_history = []

    def update_learning(self, bpm_truth, r_mean, g_mean):
        p = self.profiles.get(self.current_user)
        if not p or not p.get("trained", False): return

        # 保留原本的學習邏輯供歷史紀錄使用，但在最新的 CHROM 算法中不再強依賴它
        opt_R = r_mean / g_mean if g_mean > 0 else 3.0
        opt_R = max(2.5, min(3.5, opt_R))
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
        # 切換使用者時，自動清空歷史穩定緩衝區
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
            return {"bpm": 0, "rmssd": 0, "sdnn": 0, "error": "影像全黑 (請檢查光線)", "peaks": 0}

        # ====================================================================
        # [核心升級] 標準 CHROM 算法 (Chrominance-based method)
        # ====================================================================
        # 1. 單位化 (Normalization) - 抵抗環境光與騎車時的明暗閃爍
        mean_R, mean_G, mean_B = np.mean(R), np.mean(G), np.mean(B)
        Rn = R / mean_R if mean_R > 0 else R
        Gn = G / mean_G if mean_G > 0 else G
        Bn = B / mean_B if mean_B > 0 else B

        # 2. 投影到 X 和 Y 色度空間
        Xcomp = 3 * Rn - 2 * Gn
        Ycomp = 1.5 * Rn + Gn - 1.5 * Bn

        # 3. 動態計算自適應權重 Alpha (即時把運動造成的雜訊扣除)
        std_Y = np.std(Ycomp)
        if std_Y == 0: std_Y = 1e-6
        alpha = np.std(Xcomp) / std_Y

        # 4. 合成脈搏訊號並去趨勢
        bvp = signal.detrend(Xcomp - alpha * Ycomp)
        # ====================================================================

        # 帶通濾波 (Bandpass)：收窄至 45~150 BPM
        nyq = fps / 2.0
        low_cut = 0.75 / nyq
        high_cut = 2.5 / nyq
        peak_dist_sec = 0.4

        if is_training and polar_bpm > 30:
            target_hz = polar_bpm / 60.0
            low_cut = max(0.5, target_hz - 0.3) / nyq
            high_cut = min(3.0, target_hz + 0.3) / nyq
            peak_dist_sec = 1.0 / (target_hz + 0.5)
            if is_training_active:
                engine.update_learning(polar_bpm, np.mean(R), np.mean(G))

        # 4 階 Butterworth 濾波器
        b, a = signal.butter(4, [low_cut, high_cut], btype='band')
        f_bvp = signal.filtfilt(b, a, bvp)

        # 亞像素波峰分析 (Cubic Spline Interpolation) 上採樣至 120fps
        interp_factor = 4
        new_fps = fps * interp_factor
        t = np.arange(len(f_bvp))
        t_new = np.linspace(0, len(f_bvp) - 1, len(f_bvp) * interp_factor)
        cs = CubicSpline(t, f_bvp)
        f_bvp_interp = cs(t_new)

        peak_dist_interp = int(new_fps * peak_dist_sec)
        prominence_val = max(np.std(f_bvp_interp) * 0.15, 1e-5)
        peaks, _ = signal.find_peaks(f_bvp_interp, distance=peak_dist_interp, prominence=prominence_val)

        bpm, rmssd, sdnn = 0, 0, 0
        if len(peaks) >= 4:
            # 計算 RR 間期 (ms)
            rr = np.diff(peaks) * (1000.0 / new_fps)

            # 生理過濾器：限制在 400ms(150bpm) ~ 1500ms(40bpm)
            valid_rr = rr[(rr >= 400) & (rr <= 1500)]

            if len(valid_rr) >= 3:
                # [新增防呆機制] 剔除相鄰跳變超過 20% 的假波峰 (防運動雜訊)
                rel_diff = np.abs(np.diff(valid_rr)) / valid_rr[:-1]
                mask = np.insert(rel_diff < 0.20, 0, True)
                valid_rr = valid_rr[mask]

                if len(valid_rr) >= 2:
                    raw_bpm = 60000.0 / np.median(valid_rr)

                    # 輸出穩定化：取最近 5 次的 BPM 中位數
                    engine.bpm_history.append(raw_bpm)
                    if len(engine.bpm_history) > 5:
                        engine.bpm_history.pop(0)
                    bpm = np.median(engine.bpm_history)

                    if len(valid_rr) > 2:
                        diff_rr = np.abs(np.diff(valid_rr))
                        # RMSSD 二次防爆：超過 150ms 的跳動強制忽略
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