import numpy as np
from scipy import signal
import json


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


def process_data_from_js(rgb_data, fps, polar_bpm, profile_name, is_training):
    engine.current_user = profile_name
    buffer = np.array(rgb_data).reshape(-1, 3)

    # 將緩衝門檻降為 150 幀 (約 5 秒) 即可開始初步計算
    if len(buffer) < 150:
        return {"bpm": 0, "rmssd": 0, "sdnn": 0}

    try:
        R, G, B = buffer[:, 0], buffer[:, 1], buffer[:, 2]
        p = engine.profiles.get(profile_name, engine.profiles["Public Mode"])

        X = p["c_R"] * R - p["c_G"] * G
        Y = 1.5 * R + G - 1.5 * B

        std_Y = np.std(Y)
        if std_Y == 0: std_Y = 1e-6
        alpha = np.std(X) / std_Y
        bvp = signal.detrend(X - alpha * Y)

        nyq = fps / 2

        if is_training and polar_bpm > 30:
            target_hz = polar_bpm / 60.0
            low_cut = max(0.5, target_hz - 0.25) / nyq
            high_cut = min(3.0, target_hz + 0.25) / nyq
            peak_dist = int(fps / (target_hz + 0.5))
            engine.update_learning(polar_bpm, np.mean(R), np.mean(G))
        else:
            base_hz = p["baseline_bpm"] / 60.0
            low_cut = max(0.5, base_hz - 0.5) / nyq
            high_cut = min(3.0, base_hz + 0.5) / nyq
            peak_dist = int(fps * 0.4)

        b, a = signal.butter(4, [low_cut, high_cut], btype='band')
        f_bvp = signal.filtfilt(b, a, bvp)

        peaks, _ = signal.find_peaks(f_bvp, distance=peak_dist, prominence=np.std(f_bvp) * 0.25)

        bpm, rmssd, sdnn = 0, 0, 0
        if len(peaks) >= 5:
            rr = np.diff(peaks) * (1000.0 / fps)
            # 專業過濾：只留下生理可能的範圍 (300ms ~ 1500ms)
            valid_rr = rr[(rr > 300) & (rr < 1500)]

            if len(valid_rr) >= 3:
                # 移除突發的跳動 (中位數過濾法)
                median_rr = np.median(valid_rr)
                valid_rr = valid_rr[np.abs(valid_rr - median_rr) < (median_rr * 0.2)]

                if len(valid_rr) >= 2:
                    bpm = 60000 / np.median(valid_rr)
                    rmssd = np.sqrt(np.mean(np.diff(valid_rr) ** 2))
                    sdnn = np.std(valid_rr)

        return {"bpm": round(bpm, 1), "rmssd": round(rmssd, 1), "sdnn": round(sdnn, 1)}
    except Exception as e:
        return {"bpm": 0, "rmssd": 0, "sdnn": 0, "error": str(e)}