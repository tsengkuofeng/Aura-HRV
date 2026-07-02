import numpy as np
from scipy import signal, sparse
from scipy.sparse.linalg import spsolve
from scipy.interpolate import CubicSpline
import json
import traceback

# 重新取樣後統一使用的標準取樣率。真實攝影機擷取間隔會因裝置效能、
# 對焦/曝光鎖定、運動(騎車)等因素而抖動，若直接假設固定 fps 反推 BPM，
# 只要真實幀率偏離設定值 X%，算出來的心率就會系統性偏差 X%。
# 這是先前「心跳非常不准」最可能的主因之一。
TARGET_FPS = 30.0

# Tarvainen 平滑先驗法(smoothness priors)去趨勢的正則化強度。
# 數值越大，被視為「趨勢」的曲線越平滑，保留下來的訊號頻段越寬；
# 用合成訊號測過 10~20 之間 SNR 表現最好，取 15 當折衷值。
DETREND_LAMBDA = 15.0


def smoothness_priors_detrend(z):
    """Tarvainen, Ranta-Aho, Karjalainen (2002) 平滑先驗法去趨勢。
    比原本單純的線性 detrend 更能處理非線性的慢速漂移(如緩慢光線變化)，
    同時保留心率頻段的訊號。樣本數太少時退回線性 detrend。"""
    n = len(z)
    if n < 6:
        return signal.detrend(z)
    try:
        D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
        I = sparse.eye(n)
        A = (I + (DETREND_LAMBDA ** 2) * (D.T @ D)).tocsc()
        trend = spsolve(A, z)
        return z - trend
    except Exception:
        # 求解失敗時(理論上不太會發生)安全退回線性 detrend，不讓主流程崩潰
        return signal.detrend(z)


class WebProfileManager:
    def __init__(self):
        self.profiles = {"Public Mode": {"c_R": 3.0, "c_G": 2.0, "baseline_bpm": 75.0, "trained": False}}
        self.current_user = "Public Mode"
        # 用於穩定心跳輸出的中位數緩衝區
        self.bpm_history = []
        # 【新增】待確認候選值：偵測「連續兩次讀值彼此一致但偏離舊歷史」時，
        # 用來判斷這是心率真的變了，而不是單次雜訊
        self.pending_candidate = None

    def update_learning(self, bpm_truth, r_mean, g_mean):
        p = self.profiles.get(self.current_user)
        if not p or not p.get("trained", False): return

        # 保留原本的學習邏輯供歷史紀錄使用，但在最新的 POS 算法中不再強依賴它
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


def process_data_from_js(rgb_data, timestamps, fps, polar_bpm, profile_name, is_training, is_training_active):
    try:
        # 切換使用者時，自動清空歷史穩定緩衝區
        if engine.current_user != profile_name:
            engine.bpm_history = []
            engine.pending_candidate = None
            engine.current_user = profile_name

        if hasattr(rgb_data, 'to_py'):
            rgb_data = rgb_data.to_py()
        if hasattr(timestamps, 'to_py'):
            timestamps = timestamps.to_py()

        buffer = np.asarray(rgb_data, dtype=np.float64).reshape(-1, 3)

        if len(buffer) < 150:
            return {"bpm": 0, "rmssd": 0, "sdnn": 0, "error": "緩衝資料不足", "peaks": 0}

        # ====================================================================
        # [修正 1] 以真實 timestamp 重新取樣到均勻時間網格，修正幀率漂移
        # ====================================================================
        real_fps = fps
        ts = None
        if timestamps is not None:
            try:
                ts = np.asarray(timestamps, dtype=np.float64)
            except Exception:
                ts = None

        if ts is not None and len(ts) == len(buffer) and ts[-1] > ts[0]:
            t_raw = (ts - ts[0]) / 1000.0  # ms -> s
            duration = t_raw[-1]
            real_fps = (len(ts) - 1) / duration if duration > 0 else fps

            n_uniform = max(int(duration * TARGET_FPS), 150)
            t_uniform = np.linspace(0, duration, n_uniform)

            R = np.interp(t_uniform, t_raw, buffer[:, 0])
            G = np.interp(t_uniform, t_raw, buffer[:, 1])
            B = np.interp(t_uniform, t_raw, buffer[:, 2])
            fps = TARGET_FPS
        else:
            # 沒有 timestamp 資料時，退回舊行為 (向下相容)
            R, G, B = buffer[:, 0], buffer[:, 1], buffer[:, 2]

        if np.mean(G) < 1:
            return {"bpm": 0, "rmssd": 0, "sdnn": 0, "error": "影像全黑 (請檢查光線)", "peaks": 0}

        # ====================================================================
        # [修正 2] CHROM -> POS 演算法
        # Wang, W., den Brinker, A.C., Stuijk, S., de Haan, G. (2017)
        # "Algorithmic Principles of Remote-PPG", IEEE Trans. Biomed. Eng.
        # DOI: 10.1109/TBME.2016.2609282
        # POS 對運動雜訊(騎車、晃動)的穩健性優於 CHROM，是目前 rPPG 文獻
        # (pyVHR / rPPG-Toolbox 皆有收錄) 公認最穩健的傳統訊號方法之一。
        # ====================================================================
        mean_R, mean_G, mean_B = np.mean(R), np.mean(G), np.mean(B)
        Rn = R / mean_R if mean_R > 0 else R
        Gn = G / mean_G if mean_G > 0 else G
        Bn = B / mean_B if mean_B > 0 else B

        S1 = Gn - Bn
        S2 = -2 * Rn + Gn + Bn

        std_S2 = np.std(S2)
        if std_S2 == 0: std_S2 = 1e-6
        alpha = np.std(S1) / std_S2

        # 改用 Tarvainen 平滑先驗法取代原本單純的線性 detrend，對非線性的
        # 慢速漂移(如緩慢光線變化、非等速的微幅晃動)處理較細緻。
        bvp = smoothness_priors_detrend(S1 + alpha * S2)
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

        # ====================================================================
        # [修正 3] 訊噪比(SNR)品質門檻：頻譜能量若不集中在合理心率頻段附近，
        # 代表訊號被運動/光線雜訊淹沒，此時寧可不輸出，也不要顯示一個
        # 看似自信但其實錯誤的 BPM 數字。
        # ====================================================================
        snr_ok = True
        peak_freq = 0.0
        try:
            freqs, psd = signal.periodogram(f_bvp, fs=fps)
            band_mask = (freqs >= 0.75) & (freqs <= 2.5)
            if np.any(band_mask) and np.sum(psd[band_mask]) > 0:
                band_freqs = freqs[band_mask]
                band_psd = psd[band_mask]
                peak_freq = float(band_freqs[np.argmax(band_psd)])
                signal_mask = np.abs(band_freqs - peak_freq) <= 0.1
                signal_power = np.sum(band_psd[signal_mask])
                total_power = np.sum(band_psd)
                snr = signal_power / total_power if total_power > 0 else 0
                snr_ok = snr >= 0.25
            else:
                snr_ok = False
        except Exception:
            snr_ok = True  # SNR 檢查本身失敗時不擋主流程，避免引入新的錯誤來源

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

        if len(peaks) >= 4 and not snr_ok:
            return {
                "bpm": 0, "rmssd": 0, "sdnn": 0,
                "error": "訊號雜訊過高，請保持穩定並確保光線充足",
                "peaks": len(peaks), "real_fps": round(real_fps, 1)
            }

        bpm, rmssd, sdnn = 0, 0, 0
        if len(peaks) >= 4:
            # 計算 RR 間期 (ms)
            rr = np.diff(peaks) * (1000.0 / new_fps)

            # 生理過濾器：限制在 400ms(150bpm) ~ 1500ms(40bpm)
            valid_rr = rr[(rr >= 400) & (rr <= 1500)]

            if len(valid_rr) >= 3:
                # 剔除相鄰跳變超過 20% 的假波峰 (防運動雜訊)
                rel_diff = np.abs(np.diff(valid_rr)) / valid_rr[:-1]
                mask = np.insert(rel_diff < 0.20, 0, True)
                valid_rr = valid_rr[mask]

                if len(valid_rr) >= 2:
                    raw_bpm = 60000.0 / np.median(valid_rr)
                    fft_bpm = peak_freq * 60.0

                    # ========================================================
                    # [修正 5] 時域波峰計數 vs 頻譜(FFT)主頻率交叉驗證：
                    # 時域波峰計數在雜訊環境下，容易「穩定地」收斂到一個諧波
                    # 相關但錯誤的頻率(可能偏高、也可能偏低)——數值不會亂飄，
                    # 但就是不準。頻譜分析是對整個時間窗口做能量積分，較不受
                    # 少數幾個誤判/漏抓的波峰影響，拿來當交叉驗證基準較穩健。
                    # 兩者差距過大時：改採頻譜估計值，且不信任這次的波峰時間
                    # 點 -> 不拿來計算 HRV，避免把雜訊誤判成漂亮的 RMSSD/SDNN
                    # (這正是「明明很累，數值卻超好」的成因)。
                    # ========================================================
                    rr_reliable = True
                    if fft_bpm > 0 and abs(raw_bpm - fft_bpm) / fft_bpm > 0.12:
                        raw_bpm = fft_bpm
                        rr_reliable = False

                    # ========================================================
                    # [修正 4-v2] 生理連續性檢查（原版有嚴重 bug，現已修正）：
                    # 原本的邏輯是「與近期中位數差距超過 25% 就永久拒絕」，這會
                    # 造成一旦第一批讀值剛好誤判過高/過低，之後即使訊號回到正確
                    # 數值，也會被當成「離群值」永遠擋在外面 —— 這正是「BPM
                    # 卡在一個穩定但錯誤的數字，不會飄也不會修正」的真正原因，
                    # 已用模擬驗證重現並確認修好。
                    #
                    # 新邏輯：偏離超過 25% 時先不直接採用，而是記成「待確認候選」；
                    # 如果下一次讀值跟這個候選彼此接近（代表不是單次雜訊，是訊號
                    # 真的穩定變化了），才捨棄舊歷史、重新鎖定到新數值；如果只是
                    # 單次亂跳、下次又跟舊歷史一致，就當雜訊丟掉，不影響顯示值。
                    # ========================================================
                    accept = True
                    already_relocked = False
                    if len(engine.bpm_history) >= 3:
                        recent_median = np.median(engine.bpm_history)
                        if recent_median > 0 and abs(raw_bpm - recent_median) / recent_median > 0.25:
                            if (engine.pending_candidate is not None and
                                    abs(raw_bpm - engine.pending_candidate) / max(engine.pending_candidate, 1e-6) < 0.15):
                                # 連續兩次新讀值彼此一致 -> 心率真的變了，捨棄舊歷史重新開始
                                engine.bpm_history = [engine.pending_candidate, raw_bpm]
                                engine.pending_candidate = None
                                already_relocked = True
                            else:
                                engine.pending_candidate = raw_bpm
                                accept = False
                        else:
                            engine.pending_candidate = None

                    if accept and not already_relocked:
                        engine.bpm_history.append(raw_bpm)

                    if len(engine.bpm_history) > 5:
                        engine.bpm_history = engine.bpm_history[-5:]

                    if engine.bpm_history:
                        bpm = np.median(engine.bpm_history)

                    # HRV(RMSSD/SDNN) 只在這次的波峰時間點通過交叉驗證(rr_reliable)
                    # 時才計算，否則寧可顯示 0，也不要用不可信的波峰數據算出一個
                    # 看似漂亮、實際上只是雜訊的 HRV 數字。
                    if rr_reliable and len(valid_rr) > 2:
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
            "peaks": len(peaks),
            "real_fps": round(real_fps, 1)
        }
    except Exception as e:
        return {"bpm": 0, "rmssd": 0, "sdnn": 0, "error": str(traceback.format_exc()), "peaks": 0}