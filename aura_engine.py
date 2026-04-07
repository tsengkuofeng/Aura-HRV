import numpy as np
from scipy import signal
import json
import traceback


class WebProfileManager:
    def __init__(self):
        self.profiles = {"Public Mode": {"c_R": 3.0, "c_G": 2.0, "baseline_bpm": 75.0, "trained": False}}
        self.current_user = "Public Mode"

    def update_learning(self, bpm_truth, r_mean, g_mean):
        p = self.profiles.get(self.current_user)
        if not p or not p.get("trained", False): return

        opt_R = r_mean / g_mean if g_mean > 0 else 3.0
        p["c_R"] = (p["c_R"] * 0.95) + (opt_R * 0.05)
        p["baseline_bpm"] = (p["baseline_bpm"] * 0.9) + (bpm_truth * 0.1)


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
    engine.current_user = profile_name
    try:
        # 【修正 1】確保 Pyodide 的 JS Proxy 完美轉為 Python 陣列
        if hasattr(rgb_data, 'to_py'):
            rgb_data = rgb_data.to_py()
        buffer = np.asarray(rgb_data, dtype=np.float64).reshape(-1, 3)

        if len(buffer) < 150:
            return {"bpm": 0, "rmssd": 0, "sdnn": 0, "error": "緩衝資料不足", "peaks": 0}

        R, G, B = buffer[:, 0], buffer[:, 1], buffer[:, 2]

        # 【修正 2】防呆：如果 iOS 阻擋了 Canvas 導致全黑，提早阻斷並報錯
        if np.mean(G) < 1:
            return {"bpm": 0, "rmssd": 0, "sdnn": 0, "error": "影像全黑 (請檢查iOS隱私設定)", "peaks": 0}

        p = engine.profiles.get(profile_name, engine.profiles["Public Mode"])

        X = p["c_R"] * R - p["c_G"] * G
        Y = 1.5 * R + G - 1.5 * B

        std_Y = np.std(Y)
        if std_Y == 0: std_Y = 1e-6
        alpha = np.std(X) / std_Y
        bvp = signal.detrend(X - alpha * Y)

        nyq = fps / 2.0

        # 【修正 3】放寬濾波器範圍，不再死鎖於 baseline，容忍 45~180 BPM
        low_cut = 0.75 / nyq
        high_cut = 3.0 / nyq
        peak_dist = int(fps * 0.4)

        if is_training and polar_bpm > 30:
            target_hz = polar_bpm / 60.0
            low_cut = max(0.5, target_hz - 0.3) / nyq
            high_cut = min(3.0, target_hz + 0.3) / nyq
            peak_dist = int(fps / (target_hz + 0.5))
            if is_training_active:
                engine.update_learning(polar_bpm, np.mean(R), np.mean(G))

        b, a = signal.butter(3, [low_cut, high_cut], btype='band')
        f_bvp = signal.filtfilt(b, a, bvp)

        # 【修正 4】大幅降低波峰檢測的敏感度門檻
        prominence_val = max(np.std(f_bvp) * 0.1, 1e-5)
        peaks, _ = signal.find_peaks(f_bvp, distance=peak_dist, prominence=prominence_val)

        bpm, rmssd, sdnn = 0, 0, 0
        if len(peaks) >= 4:
            rr = np.diff(peaks) * (1000.0 / fps)
            valid_rr = rr[(rr > 300) & (rr < 1500)]

            if len(valid_rr) >= 2:
                median_rr = np.median(valid_rr)
                valid_rr = valid_rr[np.abs(valid_rr - median_rr) < (median_rr * 0.3)]

                if len(valid_rr) >= 2:
                    bpm = 60000.0 / np.median(valid_rr)
                    if len(valid_rr) > 2:
                        rmssd = np.sqrt(np.mean(np.diff(valid_rr) ** 2))
                    sdnn = np.std(valid_rr)

        return {
            "bpm": round(bpm, 1),
            "rmssd": round(rmssd, 1),
            "sdnn": round(sdnn, 1),
            "error": "",
            "peaks": len(peaks)
        }
    except Exception as e:
        # 回傳完整報錯字串給 JS
        return {"bpm": 0, "rmssd": 0, "sdnn": 0, "error": str(traceback.format_exc()), "peaks": 0}