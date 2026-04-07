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

        # 【核心修正 1】限制學習極限：
        # 強制將學習範圍鎖定在人體膚色的物理極限 (2.5 ~ 3.5 之間)
        # 避免受室內黃光或螢幕藍光干擾，導致演算法走火入魔
        opt_R = r_mean / g_mean if g_mean > 0 else 3.0
        opt_R = max(2.5, min(3.5, opt_R))

        # 【核心修正 2】降低學習速率 (原本 0.05 降為 0.01)
        # 讓光學特徵記憶更平滑，不會因為一次錯誤光線就污染數據
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
    engine.current_user = profile_name
    try:
        if hasattr(rgb_data, 'to_py'):
            rgb_data = rgb_data.to_py()
        buffer = np.asarray(rgb_data, dtype=np.float64).reshape(-1, 3)

        if len(buffer) < 150:
            return {"bpm": 0, "rmssd": 0, "sdnn": 0, "error": "緩衝資料不足", "peaks": 0}

        R, G, B = buffer[:, 0], buffer[:, 1], buffer[:, 2]

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

        prominence_val = max(np.std(f_bvp) * 0.15, 1e-5)
        peaks, _ = signal.find_peaks(f_bvp, distance=peak_dist, prominence=prominence_val)

        bpm, rmssd, sdnn = 0, 0, 0
        if len(peaks) >= 4:
            rr = np.diff(peaks) * (1000.0 / fps)
            valid_rr = rr[(rr > 300) & (rr < 1500)]

            if len(valid_rr) >= 2:
                # 中位數過濾：剔除太誇張的偏差
                median_rr = np.median(valid_rr)
                valid_rr = valid_rr[np.abs(valid_rr - median_rr) < (median_rr * 0.2)]

                if len(valid_rr) >= 2:
                    bpm = 60000.0 / np.median(valid_rr)

                    if len(valid_rr) > 2:
                        # 【核心修正 3】RMSSD 防爆機制
                        diff_rr = np.abs(np.diff(valid_rr))
                        # 過濾掉超過 150ms 的相鄰心跳跳變 (在靜止狀態下這幾乎100%是雜訊產生的假波)
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