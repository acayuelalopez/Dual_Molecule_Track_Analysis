import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from skimage import morphology, filters
from PIL import Image, ImageFilter
from decimal import Decimal
from skimage import restoration


from pandas.errors import EmptyDataError

def safe_read_csv(path, context=""):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"[SKIP] {context}: archivo no existe -> {path}")
        return pd.DataFrame()
    except EmptyDataError:
        print(f"[SKIP] {context}: CSV vacío (EmptyDataError) -> {path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"[SKIP] {context}: error leyendo CSV -> {path} | {e}")
        return pd.DataFrame()

# ========= Helper único para medias en rango (consistente en todo el pipeline) =========
def range_mean(series, idx_min, idx_max, *, fallback_to_full=True):
    """
    Devuelve la media de 'series' en el rango [idx_min : idx_max] (inclusive) usando iloc.
    - Si el rango queda vacío y fallback_to_full=True -> usa la serie completa.
    - Si la serie está vacía o todo son NaN -> devuelve NaN.

    NOTA: se asume que la 'series' ya corresponde al track ordenado por FRAME y con índice 0..N-1.
    """
    if series is None or len(series) == 0:
        return float('nan')
    if idx_min > idx_max:
        # Si el usuario invierte el rango, intercambiamos para evitar sorpresas
        idx_min, idx_max = idx_max, idx_min

    sub = series.iloc[idx_min: idx_max + 1]
    if len(sub) == 0 and fallback_to_full:
        sub = series

    # np.nanmean devuelve NaN si todo es NaN; es lo esperado
    return float(np.nanmean(sub)) if len(sub) > 0 else float('nan')


def ensure_sorted_reset(track_df):
    """
    Ordena por FRAME y resetea el índice para que iloc sea 0..N-1.
    Úsalo siempre antes de calcular medias en rango.
    """
    if track_df is None or track_df.empty:
        return track_df
    return track_df.sort_values('FRAME').reset_index(drop=True)


# =======================================================================================
def plot_overlapping_tracks(image_name, rojo_df, verde_df, file_results, plots_directory):
    for result in file_results:
        rojo_track = rojo_df[rojo_df['TRACK_ID'] == result['Rojo_TRACK_ID']]
        verde_track = verde_df[verde_df['TRACK_ID'] == result['Verde_TRACK_ID']]

        # Plot MEAN_INTENSITY_CH1 vs FRAME for Rojo with background intensity values
        plt.figure()
        plt.plot(rojo_track['FRAME'], rojo_track['MEAN_INTENSITY_CH1'], 'r-', label=f'Rojo {result["Rojo_TRACK_ID"]}')
        if 'Intensity-Bg Rojo' in rojo_track.columns:
            plt.plot(rojo_track['FRAME'], rojo_track['Intensity-Bg Rojo'], 'r--', label='Intensity-Bg Rojo')
        plt.xlabel('FRAME')
        plt.ylabel('MEAN_INTENSITY_CH1 / Intensity-Bg')
        plt.title(f'{image_name} - Rojo MEAN_INTENSITY_CH1 vs FRAME with Intensity-Bg')
        plt.legend()
        plt.savefig(os.path.join(plots_directory,
                                 f'{image_name}_Rojo_MEAN_INTENSITY_CH1_vs_FRAME_with_Intensity-Bg_{result["Rojo_TRACK_ID"]}.png'))
        plt.close()

        # Plot MEAN_INTENSITY_CH1 vs FRAME for Verde with background intensity values
        plt.figure()
        plt.plot(verde_track['FRAME'], verde_track['MEAN_INTENSITY_CH1'], 'g-',
                 label=f'Verde {result["Verde_TRACK_ID"]}')
        if 'Intensity-Bg Verde' in verde_track.columns:
            plt.plot(verde_track['FRAME'], verde_track['Intensity-Bg Verde'], 'g--', label='Intensity-Bg Verde')
        plt.xlabel('FRAME')
        plt.ylabel('MEAN_INTENSITY_CH1 / Intensity-Bg')
        plt.title(f'{image_name} - Verde MEAN_INTENSITY_CH1 vs FRAME with Intensity-Bg')
        plt.legend()
        plt.savefig(os.path.join(plots_directory,
                                 f'{image_name}_Verde_MEAN_INTENSITY_CH1_vs_FRAME_with_Intensity-Bg_{result["Verde_TRACK_ID"]}.png'))
        plt.close()


def calculate_msd(trajectory, max_lag):
    msd = np.zeros(max_lag)
    n = len(trajectory)
    for lag in range(1, max_lag + 1):
        displacements = trajectory[lag:] - trajectory[:-lag]
        squared_displacements = np.sum(displacements ** 2, axis=1)
        msd[lag - 1] = np.mean(squared_displacements)
    return msd


def calculate_alpha(msd, time_lags):
    log_msd = np.log(msd)
    log_time_lags = np.log(time_lags)
    alpha, _ = np.polyfit(log_time_lags, log_msd, 1)
    return alpha


def calculate_diffusion_coefficient(msd, time_lags):
    D = msd / (2 * time_lags)
    return D


def calculate_sMSS(trajectory_xy, max_lag, q_list=(1, 2, 3, 4, 5, 6), eps=1e-12):
    """
    Calcula sMSS como la pendiente del Moment Scaling Spectrum (MSS).
    1) M_q(tau) = < |Δr(tau)|^q >
    2) Para cada q: log M_q ~ beta(q) * log tau
    3) Ajuste lineal beta(q) ~ sMSS * q + b  -> sMSS
    """
    import numpy as np

    traj = np.asarray(trajectory_xy, dtype=float)
    n = len(traj)
    if n < 3:
        return float("nan")

    max_lag = int(max_lag)
    max_lag = min(max_lag, n - 1)
    if max_lag < 2:
        return float("nan")

    taus = np.arange(1, max_lag + 1, dtype=float)
    log_tau = np.log(taus)

    betas = []
    qs_used = []

    for q in q_list:
        Mq = []
        for tau in range(1, max_lag + 1):
            disp = traj[tau:] - traj[:-tau]
            dr = np.sqrt((disp ** 2).sum(axis=1))
            val = np.mean((dr ** q))
            Mq.append(val)

        Mq = np.asarray(Mq, dtype=float)

        # evitar log(0) y NaNs
        mask = np.isfinite(Mq) & (Mq > eps)
        if mask.sum() < 2:
            continue

        beta, _ = np.polyfit(log_tau[mask], np.log(Mq[mask]), 1)
        betas.append(beta)
        qs_used.append(q)

    if len(betas) < 2:
        return float("nan")

    betas = np.asarray(betas, dtype=float)
    qs_used = np.asarray(qs_used, dtype=float)

    sMSS, _ = np.polyfit(qs_used, betas, 1)
    return float(sMSS)


def calculate_mean(slice_np, roi):
    mean_value = np.mean(slice_np[roi])
    return mean_value


def find_csv(input_directory):
    rojo_csv_files_spots = {}
    verde_csv_files_spots = {}
    rojo_csv_files_tracks = {}
    verde_csv_files_tracks = {}

    # Iterate through the main directories
    for main_dir in os.listdir(input_directory):
        if 'rojo' in main_dir or 'verde' in main_dir:
            main_path = os.path.join(input_directory, main_dir)
            if not os.path.exists(main_path):
                continue
            # Iterate through the subdirectories
            for sub_dir in os.listdir(main_path):
                if 'Moving' in sub_dir or 'Fixed' in sub_dir or 'moving' in sub_dir or 'fixed' in sub_dir:
                    sub_path = os.path.join(main_path, sub_dir)
                    # Check if the directory contains 'SPT_Analysis'
                    spt_analysis_path = os.path.join(sub_path, 'SPT_Analysis')
                    if os.path.exists(spt_analysis_path):
                        # Find CSV files containing specific strings
                        for file_name in os.listdir(spt_analysis_path):
                            if '_Spots in tracks statistics.csv' in file_name:
                                image_name = file_name.replace('_Spots in tracks statistics.csv', '').replace('fixed',
                                                                                                              '').replace(
                                    'Fixed', '').replace('moving', '').replace('Moving', '').strip()
                                if 'rojo' in main_dir:
                                    rojo_csv_files_spots[image_name] = os.path.join(spt_analysis_path, file_name)
                                elif 'verde' in main_dir:
                                    verde_csv_files_spots[image_name] = os.path.join(spt_analysis_path, file_name)
                            elif '_Tracks statistics.csv' in file_name:
                                image_name = file_name.replace('_Tracks statistics.csv', '').replace('fixed',
                                                                                                     '').replace(
                                    'Fixed', '').replace('moving', '').replace('Moving', '').strip()
                                if 'rojo' in main_dir:
                                    rojo_csv_files_tracks[image_name] = os.path.join(spt_analysis_path, file_name)
                                elif 'verde' in main_dir:
                                    verde_csv_files_tracks[image_name] = os.path.join(spt_analysis_path, file_name)

    return rojo_csv_files_spots, verde_csv_files_spots, rojo_csv_files_tracks, verde_csv_files_tracks


def analyze_trajectories(image_name, rojo_file, verde_file,
                         distancia_x, distancia_y, min_frames_overlap, frame_gap,
                         spot_range_min, spot_range_max,
                         allowed_rojo_ids, allowed_verde_ids):
    """
    Analiza pares de trayectorias Rojo/Verde SOLO para los tracks cuyos IDs han sido
    validados previamente por threshold (allowed_rojo_ids / allowed_verde_ids).

    Nuevo comportamiento frame_gap:
    - min_frames_overlap = N: exige N frames solapantes (frames donde ROJO y VERDE están cerca)
    - frame_gap = g: permite que esos N frames no sean consecutivos, pero entre dos frames
      solapantes consecutivos se permite un salto máximo de (g + 1) frames.
        g=0 -> Δframe<=1 (consecutivo estricto)
        g=1 -> Δframe<=2 (permite 1 frame perdido)
        g=2 -> Δframe<=3 (permite 2 frames perdidos)
    """
    import numpy as np
    import pandas as pd

    print(f"Analyzing {image_name} - Rojo: {rojo_file}\n Verde: {verde_file}")

    # === Carga de CSV de spots ===
    rojo_df = pd.read_csv(rojo_file)
    verde_df = pd.read_csv(verde_file)

    # === FILTRO INICIAL: SOLO IDs PERMITIDOS ===
    rojo_df = rojo_df[rojo_df['TRACK_ID'].isin(allowed_rojo_ids)].copy()
    verde_df = verde_df[verde_df['TRACK_ID'].isin(allowed_verde_ids)].copy()

    # Si tras filtrar no queda nada para esta imagen, devolvemos resultados vacíos
    if rojo_df.empty or verde_df.empty:
        return image_name, [], rojo_df, verde_df

    # (En tu script existen, pero aquí los dejamos por si acaso)
    metrics_cache_rojo = {}
    metrics_cache_verde = {}

    # --------------------------
    # Helpers de columnas de intensidad
    # --------------------------
    def get_spot_bg_subtract_series(df_like):
        if 'Spot Intensity-Bg Subtract' in df_like.columns:
            return df_like['Spot Intensity-Bg Subtract'].astype(float)
        elif 'Intensity-Bg Subtract' in df_like.columns:
            return df_like['Intensity-Bg Subtract'].astype(float)
        else:
            raise KeyError("No encuentro 'Spot Intensity-Bg Subtract' / 'Intensity-Bg Subtract'.")

    def get_intensity_bg_series(df_like):
        # Intensity-Bg = MEAN_INTENSITY_CH1 - Spot Intensity-Bg Subtract
        spot_bg = get_spot_bg_subtract_series(df_like)
        mean_ch1 = df_like['MEAN_INTENSITY_CH1'].astype(float)
        return (mean_ch1 - spot_bg).astype(float)

    # --------------------------
    # Helper: media en rango (iloc)
    # --------------------------
    def compute_range_mean(series, idx_min, idx_max):
        """
        Media en el rango [idx_min : idx_max] (inclusive) usando iloc sobre índice reseteado.
        Si el rango está vacío, se usa toda la serie.
        """
        if series is None or len(series) == 0:
            return float('nan')
        if idx_min > idx_max:
            idx_min, idx_max = idx_max, idx_min
        sub = series.iloc[idx_min: idx_max + 1]
        if len(sub) == 0:
            sub = series
        return float(np.nanmean(sub)) if len(sub) > 0 else float('nan')

    # --------------------------
    # NUEVO Helper: seleccionar segmento válido con gaps
    # --------------------------
    def pick_best_overlap_segment(frames, min_len, frame_gap):
        """
        frames: lista ordenada de frames donde hay solape real.
        Devuelve el segmento más largo donde entre frames consecutivos:
            frames[i] - frames[i-1] <= frame_gap + 1
        y que tenga al menos min_len frames.
        """
        if not frames:
            return []

        max_step = frame_gap + 1

        best_start = 0
        best_len = 1

        start = 0
        for i in range(1, len(frames)):
            if (frames[i] - frames[i - 1]) <= max_step:
                continue

            seg_len = i - start
            if seg_len > best_len:
                best_len = seg_len
                best_start = start
            start = i

        # último segmento
        seg_len = len(frames) - start
        if seg_len > best_len:
            best_len = seg_len
            best_start = start

        best_segment = frames[best_start: best_start + best_len]
        return best_segment if len(best_segment) >= min_len else []

    # --------------------------
    # Helper: calcular métricas de track (sobre el DF que se le pase)
    # --------------------------
    def compute_track_metrics(track_df, cache_dict):
        tid = int(track_df['TRACK_ID'].iloc[0])
        if tid in cache_dict:
            return cache_dict[tid]

        track_df = ensure_sorted_reset(track_df)
        traj = track_df[['POSITION_X', 'POSITION_Y']].values
        n = len(traj)

        # Elegir max_lag más estable para métricas de pendiente (alpha / sMSS)
        # - si la trayectoria es corta, no tiene sentido pedir muchos lags
        # - si es más larga, usar más lags mejora estabilidad
        if n < 8:
            max_lag = min(4, n - 1)
        else:
            max_lag = min(10, n - 1, max(4, n // 4))

        time_lags = np.arange(1, max_lag + 1)

        msd = calculate_msd(traj, max_lag)
        msd = np.abs(msd)

        # alpha (requiere >=2 puntos)
        try:
            alpha = float(abs(calculate_alpha(msd, time_lags))) if len(time_lags) >= 2 else float('nan')
        except Exception:
            alpha = float('nan')

        # sMSS (MSS slope) - más apropiado para trayectorias largas; devolverá nan si no hay puntos
        try:
            sMSS = float(calculate_sMSS(traj, max_lag))
        except Exception:
            sMSS = float('nan')

        # D lag=1 (sin polyfit con 1 punto)
        D1 = float(msd[0]) if len(msd) >= 1 else 0.0

        valid_pts = min(4, len(time_lags), len(msd))
        if valid_pts >= 2:
            slope, _ = np.polyfit(time_lags[:valid_pts], msd[:valid_pts], 1)
            D1_4 = float(abs(slope) / valid_pts)  # tu convención
        else:
            D1_4 = 0.0

        spot_bg_series = get_spot_bg_subtract_series(track_df)
        int_bg_series = get_intensity_bg_series(track_df)

        spot_bg_mean = float(np.nanmean(spot_bg_series)) if len(spot_bg_series) else float('nan')
        int_bg_mean = float(np.nanmean(int_bg_series)) if len(int_bg_series) else float('nan')

        spot_bg_range = compute_range_mean(spot_bg_series, spot_range_min, spot_range_max)
        int_bg_range = compute_range_mean(int_bg_series, spot_range_min, spot_range_max)

        info = {
            'msd': msd,
            'alpha': alpha,
            'sMSS': sMSS,
            'D1': D1,
            'D1_4': D1_4,
            'spot_bg_mean': spot_bg_mean,
            'int_bg_mean': int_bg_mean,
            'spot_bg_range': spot_bg_range,
            'int_bg_range': int_bg_range,
        }
        cache_dict[tid] = info
        return info

    # =======================================================================================
    # Bucle por pares de TRACK_ID (solo IDs permitidos, ya filtrados arriba)
    # =======================================================================================
    file_results = []

    for rojo_track_id in sorted(rojo_df['TRACK_ID'].unique()):
        rojo_track = ensure_sorted_reset(rojo_df[rojo_df['TRACK_ID'] == rojo_track_id])

        for verde_track_id in sorted(verde_df['TRACK_ID'].unique()):
            verde_track = ensure_sorted_reset(verde_df[verde_df['TRACK_ID'] == verde_track_id])

            # ------------------------------------------------------------
            # 1) Encuentra frames donde hay solape espacial REAL (mismo FRAME)
            # ------------------------------------------------------------
            m = rojo_track.merge(
                verde_track,
                on='FRAME',
                suffixes=('_Rojo', '_Verde')
            )
            m = m[(m['POSITION_X_Rojo'] - m['POSITION_X_Verde']).abs() <= distancia_x]
            m = m[(m['POSITION_Y_Rojo'] - m['POSITION_Y_Verde']).abs() <= distancia_y]

            # Salida temprana: si no hay frames solapantes, este par no cuenta
            if m.empty:
                continue

            # Todos los frames con solape real
            frames_all = sorted(m['FRAME'].unique())

            # ------------------------------------------------------------
            # 2) APLICA frame_gap: elige el segmento más largo que cumpla gaps
            # ------------------------------------------------------------
            frames_sel = pick_best_overlap_segment(frames_all, min_frames_overlap, frame_gap)
            if not frames_sel:
                # No hay ningún segmento que llegue a min_frames_overlap con gaps permitidos
                continue

            # Usaremos SOLO el segmento seleccionado para recalcular métricas e intensidades
            overlap_frames_list = frames_sel

            # Start/end del segmento usado
            overlap_start_frame_rojo = int(overlap_frames_list[0])
            overlap_end_frame_rojo = int(overlap_frames_list[-1])
            overlap_start_frame_verde = overlap_start_frame_rojo
            overlap_end_frame_verde = overlap_end_frame_rojo

            overlap_frames_rojo = len(overlap_frames_list)
            overlap_frames_verde = overlap_frames_rojo

            # (Opcional recomendado) "Real total" sin segmentar (para trazabilidad)
            overlap_real_frames = len(frames_all)
            overlap_real_start_frame = int(frames_all[0]) if frames_all else np.nan
            overlap_real_end_frame = int(frames_all[-1]) if frames_all else np.nan

            # ==========================================================
            # 3) Construir solape completo vs solape en rango usuario
            # ==========================================================
            # Ahora overlap_frames_list YA ES el segmento válido (con gaps)
            # Rango dentro del solape: si no hay suficientes frames, usar TODO el solape
            if len(overlap_frames_list) > spot_range_max:
                overlap_range_frames = overlap_frames_list[spot_range_min: spot_range_max + 1]
            else:
                overlap_range_frames = overlap_frames_list

            overlap_full_frames = len(overlap_frames_list)
            overlap_full_start_frame = int(overlap_frames_list[0]) if overlap_full_frames else np.nan
            overlap_full_end_frame = int(overlap_frames_list[-1]) if overlap_full_frames else np.nan

            overlap_used_frames = len(overlap_range_frames)
            overlap_used_start_frame = int(overlap_range_frames[0]) if overlap_used_frames else np.nan
            overlap_used_end_frame = int(overlap_range_frames[-1]) if overlap_used_frames else np.nan

            # DataFrames recortados
            rojo_overlap_full = ensure_sorted_reset(
                rojo_track[rojo_track['FRAME'].isin(overlap_frames_list)].copy()
            )
            verde_overlap_full = ensure_sorted_reset(
                verde_track[verde_track['FRAME'].isin(overlap_frames_list)].copy()
            )
            rojo_overlap_range = ensure_sorted_reset(
                rojo_track[rojo_track['FRAME'].isin(overlap_range_frames)].copy()
            )
            verde_overlap_range = ensure_sorted_reset(
                verde_track[verde_track['FRAME'].isin(overlap_range_frames)].copy()
            )

            def safe_nanmean(x):
                return float(np.nanmean(x)) if x is not None and len(x) else float('nan')

            # Intensidades sobre full solape y sobre rango
            spot_bg_rojo_full = safe_nanmean(
                get_spot_bg_subtract_series(rojo_overlap_full)) if not rojo_overlap_full.empty else np.nan
            int_bg_rojo_full = safe_nanmean(
                get_intensity_bg_series(rojo_overlap_full)) if not rojo_overlap_full.empty else np.nan
            spot_bg_rojo_range = safe_nanmean(
                get_spot_bg_subtract_series(rojo_overlap_range)) if not rojo_overlap_range.empty else np.nan
            int_bg_rojo_range = safe_nanmean(
                get_intensity_bg_series(rojo_overlap_range)) if not rojo_overlap_range.empty else np.nan

            spot_bg_verde_full = safe_nanmean(
                get_spot_bg_subtract_series(verde_overlap_full)) if not verde_overlap_full.empty else np.nan
            int_bg_verde_full = safe_nanmean(
                get_intensity_bg_series(verde_overlap_full)) if not verde_overlap_full.empty else np.nan
            spot_bg_verde_range = safe_nanmean(
                get_spot_bg_subtract_series(verde_overlap_range)) if not verde_overlap_range.empty else np.nan
            int_bg_verde_range = safe_nanmean(
                get_intensity_bg_series(verde_overlap_range)) if not verde_overlap_range.empty else np.nan

            # ==========================================================
            # 4) Cálculo de métricas SOLO sobre el solape (full)
            # ==========================================================
            # IMPORTANTE: no usar cache por TRACK_ID porque cambia según el par.
            r = compute_track_metrics(rojo_overlap_full, {})
            v = compute_track_metrics(verde_overlap_full, {})

            msd_rojo = r['msd']
            msd_verde = v['msd']
            alpha_rojo = r['alpha']
            alpha_verde = v['alpha']
            sMSS_rojo = r['sMSS']
            sMSS_verde = v['sMSS']
            D_rojo_slope = r['D1']
            D_verde_slope = v['D1']
            D1_4_rojo_slope = r['D1_4']
            D1_4_verde_slope = v['D1_4']

            # SIN rango (todo el solape full)
            spot_intensity_bg_rojo_mean = float(spot_bg_rojo_full)
            intensity_bg_rojo_mean = float(int_bg_rojo_full)
            spot_intensity_bg_verde_mean = float(spot_bg_verde_full)
            intensity_bg_verde_mean = float(int_bg_verde_full)

            # CON rango (sub-rango dentro del solape)
            spot_intensity_bg_subtract_range_rojo = float(spot_bg_rojo_range)
            intensity_bg_range_rojo = float(int_bg_rojo_range)
            spot_intensity_bg_subtract_range_verde = float(spot_bg_verde_range)
            intensity_bg_range_verde = float(int_bg_verde_range)

            # (Mantengo también tu forma previa: usando compute_range_mean sobre la serie del solape full)
            # pero en tu script estaba duplicado; esto lo dejo coherente con tu estructura:
            spot_intensity_bg_subtract_range_rojo = r['spot_bg_range']
            spot_intensity_bg_subtract_range_verde = v['spot_bg_range']
            intensity_bg_range_rojo = r['int_bg_range']
            intensity_bg_range_verde = v['int_bg_range']

            # Media MEAN_INTENSITY en el solape
            mean_intensity_rojo = float(
                rojo_overlap_full[[c for c in rojo_overlap_full.columns if 'MEAN_INTENSITY_' in c][0]].mean()
            ) if not rojo_overlap_full.empty else np.nan
            mean_intensity_verde = float(
                verde_overlap_full[[c for c in verde_overlap_full.columns if 'MEAN_INTENSITY_' in c][0]].mean()
            ) if not verde_overlap_full.empty else np.nan

            # Longitud para clasificación: usar FRAMES SOLAPADOS (Overlap_Frames_*)
            # (Estos valores son los que luego se guardan como 'Overlap_Frames_Rojo/Verde' en el CSV)
            rojo_length_for_class = overlap_frames_rojo
            verde_length_for_class = overlap_frames_verde

            rojo_length_category = 'Long' if rojo_length_for_class >= length_threshold else 'Short'
            verde_length_category = 'Long' if verde_length_for_class >= length_threshold else 'Short'


            # Motilidad por D1-4
            rojo_motility = 'Mobile' if D1_4_rojo_slope >= motility_threshold else 'Inmobile'
            verde_motility = 'Mobile' if D1_4_verde_slope >= motility_threshold else 'Inmobile'

            def classify_smss(smss_val, tol=1e-6):
                import numpy as np
                if smss_val is None or not np.isfinite(smss_val):
                    return 'Undefined'
                if abs(smss_val - 0.0) <= tol:
                    return 'Immobility'
                # Banda "Free" alrededor de 0.5 (coherente con TrackAnalyzer; ajustable si lo necesitas)
                if smss_val < 0.45:
                    return 'Confined'
                elif 0.45 <= smss_val <= 0.55:
                    return 'Free'
                elif smss_val > 0.55:
                    return 'Directed'
                return 'Undefined'

            def classify_alpha(alpha_val):
                if 0 <= alpha_val < 0.6:
                    return 'Confined'
                elif 0.6 <= alpha_val < 0.9:
                    return 'Anomalous'
                elif 0.9 <= alpha_val < 1.1:
                    return 'Free'
                elif alpha_val >= 1.1:
                    return 'Directed'
                return 'Undefined'

            rojo_movement = classify_smss(sMSS_rojo)
            verde_movement = classify_smss(sMSS_verde)
            alpha_movement_rojo = classify_alpha(alpha_rojo)
            alpha_movement_verde = classify_alpha(alpha_verde)

            # ======================
            # 5) Registro de resultados
            # ======================
            file_results.append({
                'Image Title': image_name,
                'Rojo_TRACK_ID': rojo_track_id,
                'Verde_TRACK_ID': verde_track_id,

                'Track_Total_Frames_Rojo': rojo_track['FRAME'].nunique(),
                'Track_Total_Frames_Verde': verde_track['FRAME'].nunique(),

                'Total_Overlap_Segment_Frames_Rojo': overlap_frames_rojo,
                'Total_Overlap_Segment_Frames_Verde': overlap_frames_verde,
                'Overlap_Segment_Start_Frame_Rojo': overlap_start_frame_rojo,
                'Overlap_Segment_End_Frame_Rojo': overlap_end_frame_rojo,
                'Overlap_Segment_Start_Frame_Verde': overlap_start_frame_verde,
                'Overlap_Segment_End_Frame_Verde': overlap_end_frame_verde,

                # Segmento "full" usado (con gaps) (equivale a overlap_frames_list)
                'Total_Overlap_Segment_Frames': overlap_full_frames,
                'Overlap_Segment_Start_Frame': overlap_full_start_frame,
                'Overlap_Segment_End_Frame': overlap_full_end_frame,


                # "Real total" sin segmentar (todos los frames con solape espacial)
                'Total_Overlap_Frames_Real': overlap_real_frames,
                'Overlap_Frames_Real_Start_Frame': overlap_real_start_frame,
                'Overlap_Frames_Real_End_Frame': overlap_real_end_frame,

                # Segmento usado (con gaps) y rango usado para recálculo
                'Total_Overlap_Used_Frames_For_Recalc': overlap_used_frames,
                'Overlap_Used_Start_Frame': overlap_used_start_frame,
                'Overlap_Used_End_Frame': overlap_used_end_frame,

                'MSD Time Lag 1 Rojo': float(msd_rojo[0]) if len(msd_rojo) > 0 else np.nan,
                'MSD Time Lag 2 Rojo': float(msd_rojo[1]) if len(msd_rojo) > 1 else np.nan,
                'MSD Time Lag 3 Rojo': float(msd_rojo[2]) if len(msd_rojo) > 2 else np.nan,
                'MSD Time Lag 1 Verde': float(msd_verde[0]) if len(msd_verde) > 0 else np.nan,
                'MSD Time Lag 2 Verde': float(msd_verde[1]) if len(msd_verde) > 1 else np.nan,
                'MSD Time Lag 3 Verde': float(msd_verde[2]) if len(msd_verde) > 2 else np.nan,

                'MSD Rojo': float(msd_rojo[min(3, len(msd_rojo) - 1)]) if len(msd_rojo) else float('nan'),
                'MSD Verde': float(msd_verde[min(3, len(msd_verde) - 1)]) if len(msd_verde) else float('nan'),

                'Diffusion Coefficient Rojo': float(D_rojo_slope),
                'Diffusion Coefficient Verde': float(D_verde_slope),

                'Alpha Rojo': float(alpha_rojo),
                'Alpha Verde': float(alpha_verde),
                'Alpha Movement Rojo': alpha_movement_rojo,
                'Alpha Movement Verde': alpha_movement_verde,

                'sMSS Rojo': float(sMSS_rojo),
                'sMSS Verde': float(sMSS_verde),
                'sMSS Rojo Movement': rojo_movement,
                'sMSS Verde Movement': verde_movement,

                'Short-Time Lag Diffusion Coefficient Rojo (D1-4)': float(D1_4_rojo_slope),
                'Short-Time Lag Diffusion Coefficient Verde (D1-4)': float(D1_4_verde_slope),

                'Track Mean Intensity in Spot Range (' + str(spot_range_min) + '-' + str(
                    spot_range_max) + ') Rojo': mean_intensity_rojo,
                'Track Mean Intensity in Spot Range (' + str(spot_range_min) + '-' + str(
                    spot_range_max) + ') Verde': mean_intensity_verde,

                'Track Length Rojo': rojo_length_category,
                'Track Length Verde': verde_length_category,
                'Motility Rojo': rojo_motility,
                'Motility Verde': verde_motility,

                'Spot Intensity-Bg Rojo': spot_intensity_bg_rojo_mean,
                'Intensity-Bg Rojo': intensity_bg_rojo_mean,
                'Spot Intensity-Bg Verde': spot_intensity_bg_verde_mean,
                'Intensity-Bg Verde': intensity_bg_verde_mean,

                'Spot Intensity-Bg (' + str(spot_range_min) + '-' + str(
                    spot_range_max) + ') Rojo': spot_intensity_bg_subtract_range_rojo,
                'Intensity-Bg (' + str(spot_range_min) + '-' + str(spot_range_max) + ') Rojo': intensity_bg_range_rojo,
                'Spot Intensity-Bg (' + str(spot_range_min) + '-' + str(
                    spot_range_max) + ') Verde': spot_intensity_bg_subtract_range_verde,
                'Intensity-Bg (' + str(spot_range_min) + '-' + str(
                    spot_range_max) + ') Verde': intensity_bg_range_verde,




            })

    # ---------------------------------------------------------
    # Refiltrado por umbral de rango (como ya hacías al final)
    # ---------------------------------------------------------
    results_df = pd.DataFrame(file_results)

    range_col_r = f"Intensity-Bg ({spot_range_min}-{spot_range_max}) Rojo"
    range_col_v = f"Intensity-Bg ({spot_range_min}-{spot_range_max}) Verde"

    if (not results_df.empty) and (range_col_r in results_df.columns) and (range_col_v in results_df.columns):
        before = len(results_df)
        results_df = results_df[
            (results_df[range_col_r] >= thr_intensity_bg_range_rojo) &
            (results_df[range_col_v] >= thr_intensity_bg_range_verde)
            ].copy()
        print(f"[THR-ANALYZE] {image_name}: {before}->{len(results_df)} filas tras refiltrar por rango/umbral.")
        file_results = results_df.to_dict(orient='records')

    return image_name, file_results, rojo_df, verde_df


# Ask the user for the input directory
input_directory = input("Please enter the input directory: ")

# Ask the user for additional parameters with default values
distancia_x = float(input("Please enter the distance in X (default 0.4): ") or 0.4)
distancia_y = float(input("Please enter the distance in Y (default 0.4): ") or 0.4)
min_frames_overlap = int(
    input("Please enter the minimum number of overlapping frames in the trajectory (default 5): ") or 5)
frame_gap = int(input("Please enter the max number of frames per gap (default 0): ") or 0)
spot_range = input("Please enter the spot range (e.g., 0-19) (default 0-19): ") or "0-19"
spot_range_min, spot_range_max = map(int, spot_range.split('-'))
length_threshold = int(input("Please enter the length threshold for long tracks (default 30): ") or 30)
motility_threshold = float(input("Please enter the D1-4 motility threshold (default 0.002): ") or 0.002)


# Thresholds para filtrar trayectorias por intensidad corregida de fondo (media en el rango de spot)
def parse_float(user_input, default):
    s = (user_input or "").strip().replace(",", ".")
    try:
        return float(s) if s else default
    except ValueError:
        print(f"[WARN] Valor no válido '{user_input}'. Se usa por defecto {default}.")
        return default


thr_intensity_bg_range_rojo = parse_float(
    input("Threshold Intensity-Bg range ROJO (default 33.0): "), 33.0
)
thr_intensity_bg_range_verde = parse_float(
    input("Threshold Intensity-Bg range VERDE (default 33.0): "), 33.0
)
print(
    f"[THR] ROJO={thr_intensity_bg_range_rojo}, VERDE={thr_intensity_bg_range_verde}, rango={spot_range_min}-{spot_range_max}")
# Get the column indexes for summary from the user
column_indexes_no_overlap_input = input(
    "Get the column indexes (numeric numbers from 0) for No Overlapping summary (default 33,35): ")
column_indexes_overlap_input_rojo = input("Get the column indexes for Overlapping summary ROJO (default 29,41): ")
column_indexes_overlap_input_verde = input("Get the column indexes for Overlapping summary VERDE (default 30,43): ")

if column_indexes_no_overlap_input:
    column_indexes_no_overlap = list(map(int, column_indexes_no_overlap_input.split(',')))
else:
    column_indexes_no_overlap = [33, 35]

if column_indexes_overlap_input_rojo:
    column_indexes_overlap_rojo = list(map(int, column_indexes_overlap_input_rojo.split(',')))
else:
    column_indexes_overlap_rojo = [29, 41]

if column_indexes_overlap_input_verde:
    column_indexes_overlap_verde = list(map(int, column_indexes_overlap_input_verde.split(',')))
else:
    column_indexes_overlap_verde = [30, 43]

# Find the CSV and TIF files
rojo_csv_files_spots, verde_csv_files_spots, rojo_csv_files_tracks, verde_csv_files_tracks = find_csv(
    input_directory)


def get_background_series(df_like):
    if 'Spot Intensity-Bg Subtract' in df_like.columns:
        return df_like['Spot Intensity-Bg Subtract'].astype(float)
    elif 'Intensity-Bg Subtract' in df_like.columns:
        return df_like['Intensity-Bg Subtract'].astype(float)
    else:
        raise KeyError("No encuentro la columna de fondo.")


def get_spot_bg_subtract_series(df_like):
    # Prioriza la columna ya renombrada; si aún no existe, usa la original
    if 'Spot Intensity-Bg Subtract' in df_like.columns:
        return df_like['Spot Intensity-Bg Subtract'].astype(float)
    elif 'Intensity-Bg Subtract' in df_like.columns:
        return df_like['Intensity-Bg Subtract'].astype(float)
    else:
        raise KeyError("No encuentro la columna de 'Spot Intensity-Bg Subtract' / 'Intensity-Bg Subtract'.")


def get_intensity_bg_series(df_like):
    # Intensity-Bg = MEAN_INTENSITY_CH1 - Spot Intensity-Bg Subtract
    spot_bg = get_spot_bg_subtract_series(df_like)
    mean_ch1 = df_like['MEAN_INTENSITY_CH1'].astype(float)
    return (mean_ch1 - spot_bg).astype(float)


def allowed_ids_from_spots(spots_csv_path, thr, idx_min, idx_max):
    import numpy as np, pandas as pd
    df = pd.read_csv(spots_csv_path)
    allowed = set()
    for tid, g in df.groupby('TRACK_ID'):
        g = ensure_sorted_reset(g)
        if len(g) <= idx_max:
            continue
        series_fondo = get_background_series(g)
        mval = range_mean(series_fondo, idx_min, idx_max, fallback_to_full=False)
        if np.isfinite(mval) and (mval >= thr):
            allowed.add(tid)
    return allowed


# Calcula y guarda los IDs permitidos por imagen y por color
allowed_ids_rojo_by_image = {}
allowed_ids_verde_by_image = {}

for image_name in sorted(set(rojo_csv_files_spots) & set(verde_csv_files_spots)):
    allowed_r = allowed_ids_from_spots(
        rojo_csv_files_spots[image_name],
        thr_intensity_bg_range_rojo,
        spot_range_min, spot_range_max
    )
    allowed_v = allowed_ids_from_spots(
        verde_csv_files_spots[image_name],
        thr_intensity_bg_range_verde,
        spot_range_min, spot_range_max
    )
    allowed_ids_rojo_by_image[image_name] = allowed_r
    allowed_ids_verde_by_image[image_name] = allowed_v
# ===== FIN PREFILTRO =====
# Print keys for debugging
# print("Keys in rojo_csv_files_spots:", list(rojo_csv_files_spots.keys()))
# print("Keys in rojo_tif_files:", list(rojo_tif_files.keys()))

# === Analyze the trajectories in parallel and get the results (solo IDs permitidos) ===
results_spots = {}
rojo_dfs_spots = {}
verde_dfs_spots = {}

with ThreadPoolExecutor() as executor:
    futures_spots = []
    # Recorremos solo imágenes presentes en ambos colores
    for image_name in sorted(rojo_csv_files_spots.keys()):
        if image_name not in verde_csv_files_spots:
            continue

        # Recuperar conjuntos de IDs permitidos calculados en el prefiltro
        allowed_r = allowed_ids_rojo_by_image.get(image_name, set())
        allowed_v = allowed_ids_verde_by_image.get(image_name, set())

        # Si no hay tracks válidos en alguno de los colores, saltamos la imagen
        if not allowed_r or not allowed_v:
            print(f"[SKIP] {image_name}: no hay tracks válidos (rojo={len(allowed_r)}, verde={len(allowed_v)})")
            continue

        # Encolar tarea con los IDs permitidos
        futures_spots.append(
            executor.submit(
                analyze_trajectories,
                image_name,
                rojo_csv_files_spots[image_name],
                verde_csv_files_spots[image_name],
                distancia_x, distancia_y, min_frames_overlap, frame_gap,
                spot_range_min, spot_range_max,
                allowed_r, allowed_v  # <<< IDs permitidos por color
            )
        )

    # Recoger resultados
    for fut in futures_spots:
        try:
            image_name, file_results, rojo_df, verde_df = fut.result()
        except Exception as e:
            import traceback

            print(f"[ERROR] Falló analyze_trajectories: {e}")
            traceback.print_exc()
            continue

        # Si tras el análisis no hay resultados (p.ej. no cumplieron solape mínimo), informar y seguir
        if not file_results or (rojo_df.empty and verde_df.empty):
            print(f"[WARN] {image_name}: sin resultados tras filtrar/analizar.")
            continue

        results_spots[image_name] = file_results
        rojo_dfs_spots[image_name] = rojo_df
        verde_dfs_spots[image_name] = verde_df

# Mensaje informativo si no se procesó ninguna imagen
if not results_spots:
    print("[INFO] No se encontraron imágenes con tracks válidos que cumplan los thresholds.")

# Create the results_dualTIRFM directory in input_directory and rewrite if already exists
results_directory = os.path.join(input_directory, 'results_dualTIRFM')
if os.path.exists(results_directory):
    import shutil

    shutil.rmtree(results_directory)
os.makedirs(results_directory)

# Create the 'Summary_Analysis' directory in the results_directory
summary_analysis_directory = os.path.join(results_directory, 'Summary_Analysis')
os.makedirs(summary_analysis_directory, exist_ok=True)
print(f"Directory 'Summary_Analysis' created successfully in {results_directory}.")


def extract_column_values_single_file(column_indexes_no_overlap,
                                      column_indexes_overlap_rojo,
                                      column_indexes_overlap_verde):
    """
    Extrae columnas (por índice) desde los CSV YA FILTRADOS por threshold:
      - No overlap:  <results>/<imagen>/csv/<imagen>_rojo_No_overlap_Track_statistics.csv
                    <results>/<imagen>/csv/<imagen>_verde_No_overlap_Track_statistics.csv
      - Overlap:     <results>/<imagen>/csv/<imagen>_recalculated_trajectory_overlap_results.csv
    y genera:
      - Archivos por columna (ROJO/VERDE; overlap/no-overlap) apilando cada imagen como una columna.
      - Versiones '1 columna' con filas concatenadas y una línea en blanco entre imágenes.

    NOTAS:
      - Los índices 'column_indexes_*' se aplican a los CSV PROCESADOS, no a los originales.
      - Si un índice está fuera de rango, se avisa y se omite esa columna para esa imagen.
    """
    import os
    import pandas as pd
    import numpy as np

    os.makedirs(summary_analysis_directory, exist_ok=True)

    # Diccionarios de acumulación por índice -> lista de listas (una lista por imagen)
    column_values_dict_rojo_no_overlap = {idx: [] for idx in column_indexes_no_overlap}
    column_values_dict_verde_no_overlap = {idx: [] for idx in column_indexes_no_overlap}
    column_values_dict_rojo_overlap = {idx: [] for idx in column_indexes_overlap_rojo}
    column_values_dict_verde_overlap = {idx: [] for idx in column_indexes_overlap_verde}

    # Para nombres de columnas "bonitos" por índice (se fija en la 1ª imagen válida)
    no_overlap_colnames = {}  # {idx: nombre_columna}
    overlap_colnames_r = {}  # {idx: nombre_columna} para ROJO (mismo results_df)
    overlap_colnames_v = {}  # {idx: nombre_columna} para VERDE (mismo results_df)

    image_names = []

    # Helpers
    def safe_get_column_series_by_index(df, idx):
        """Devuelve (serie, nombre_columna) si idx es válido; si no, (None, None)."""
        if df is None or df.empty:
            return None, None
        if not isinstance(idx, int):
            return None, None
        if idx < 0 or idx >= df.shape[1]:
            return None, None
        colname = df.columns[idx]
        return df.iloc[:, idx], colname

    def flatten_with_blanks(list_of_lists):
        """Aplana [[...], [...], ...] e inserta '' al final de cada sublista."""
        flat = []
        for sub in list_of_lists:
            flat.extend(sub if isinstance(sub, list) else [])
            flat.append("")  # línea en blanco separadora
        return flat

    # Iteramos por imágenes presentes en ambos colores (orden estable)
    for image_name in sorted(rojo_csv_files_tracks.keys()):
        if image_name not in verde_csv_files_tracks:
            continue

        csv_directory = os.path.join(results_directory, image_name, "csv")

        # --- Cargas de los CSV PROCESADOS (ya filtrados por threshold) ---
        try:
            rojo_tracks_no_overlap_path = os.path.join(csv_directory,
                                                       f"{image_name}_rojo_No_overlap_Track_statistics.csv")
            verde_tracks_no_overlap_path = os.path.join(csv_directory,
                                                        f"{image_name}_verde_No_overlap_Track_statistics.csv")
            results_overlap_path = os.path.join(csv_directory,
                                                f"{image_name}_recalculated_trajectory_overlap_results.csv")

            if not (os.path.exists(rojo_tracks_no_overlap_path) and
                    os.path.exists(verde_tracks_no_overlap_path) and
                    os.path.exists(results_overlap_path)):
                print(f"[WARN] {image_name}: faltan uno o más CSV procesados; se omite en el resumen.")
                continue

            rojo_tracks_df_no_ov = pd.read_csv(rojo_tracks_no_overlap_path)
            verde_tracks_df_no_ov = pd.read_csv(verde_tracks_no_overlap_path)
            results_df = pd.read_csv(results_overlap_path)

        except Exception as e:
            import traceback
            print(f"[ERROR] {image_name}: error leyendo CSV procesados -> {e}")
            traceback.print_exc()
            continue

        # Si todos vacíos, no tiene sentido agregar esta imagen
        if (rojo_tracks_df_no_ov is None or verde_tracks_df_no_ov is None or results_df is None or
                (rojo_tracks_df_no_ov.empty and verde_tracks_df_no_ov.empty and results_df.empty)):
            print(f"[INFO] {image_name}: CSVs vacíos; se omite en el resumen.")
            continue

        # ---------------------------------------------------------
        # NO OVERLAP: extrae columnas por índice (ROJO y VERDE)
        # ---------------------------------------------------------
        for idx in column_indexes_no_overlap:
            # ROJO
            serie_r, name_r = safe_get_column_series_by_index(rojo_tracks_df_no_ov, idx)
            if serie_r is not None:
                column_values_dict_rojo_no_overlap[idx].append(serie_r.values.tolist())
                if idx not in no_overlap_colnames:
                    no_overlap_colnames[idx] = str(name_r)
            else:
                # si no hay columna válida, agrega lista vacía para mantener alineación de imágenes
                column_values_dict_rojo_no_overlap[idx].append([])
                if idx not in no_overlap_colnames:
                    no_overlap_colnames[idx] = f"col_{idx}"

            # VERDE
            serie_v, name_v = safe_get_column_series_by_index(verde_tracks_df_no_ov, idx)
            if serie_v is not None:
                column_values_dict_verde_no_overlap[idx].append(serie_v.values.tolist())
                # Preferimos el nombre de ROJO si ya existe; si no, fijamos el de VERDE
                if idx not in no_overlap_colnames or no_overlap_colnames[idx].startswith("col_"):
                    no_overlap_colnames[idx] = str(name_v) if name_v is not None else no_overlap_colnames[idx]
            else:
                column_values_dict_verde_no_overlap[idx].append([])
                if idx not in no_overlap_colnames:
                    no_overlap_colnames[idx] = f"col_{idx}"

        # ---------------------------------------------------------
        # OVERLAP: extrae columnas por índice desde results_df
        #   - Para ROJO y VERDE usamos el mismo results_df (pares overlap)
        # ---------------------------------------------------------
        for idx in column_indexes_overlap_rojo:
            serie, name = safe_get_column_series_by_index(results_df, idx)
            if serie is not None:
                column_values_dict_rojo_overlap[idx].append(serie.values.tolist())
                if idx not in overlap_colnames_r:
                    overlap_colnames_r[idx] = str(name)
            else:
                column_values_dict_rojo_overlap[idx].append([])
                if idx not in overlap_colnames_r:
                    overlap_colnames_r[idx] = f"col_{idx}"

        for idx in column_indexes_overlap_verde:
            serie, name = safe_get_column_series_by_index(results_df, idx)
            if serie is not None:
                column_values_dict_verde_overlap[idx].append(serie.values.tolist())
                if idx not in overlap_colnames_v:
                    overlap_colnames_v[idx] = str(name)
            else:
                column_values_dict_verde_overlap[idx].append([])
                if idx not in overlap_colnames_v:
                    overlap_colnames_v[idx] = f"col_{idx}"

        image_names.append(image_name)

    # ==========================================================
    # GUARDADOS: No overlap (ROJO y VERDE) como matrices imagen-columna
    # ==========================================================
    for idx in column_indexes_no_overlap:
        colname = no_overlap_colnames.get(idx, f"col_{idx}")

        # --- ROJO No overlap ---
        df_r_no_ov = pd.DataFrame(column_values_dict_rojo_no_overlap[idx]).transpose()
        df_r_no_ov.columns = image_names
        out_r_no_ov = os.path.join(
            summary_analysis_directory,
            f"{colname}_rojo_No_overlap_Track_statistics.csv"
        )
        df_r_no_ov.to_csv(out_r_no_ov, index=False)

        # --- VERDE No overlap ---
        df_v_no_ov = pd.DataFrame(column_values_dict_verde_no_overlap[idx]).transpose()
        df_v_no_ov.columns = image_names
        out_v_no_ov = os.path.join(
            summary_analysis_directory,
            f"{colname}_verde_No_overlap_Track_statistics.csv"
        )
        df_v_no_ov.to_csv(out_v_no_ov, index=False)

    # ==========================================================
    # GUARDADOS: Overlap (ROJO y VERDE) como matrices imagen-columna
    # ==========================================================
    for idx in column_indexes_overlap_rojo:
        colname = overlap_colnames_r.get(idx, f"col_{idx}")
        df_r_ov = pd.DataFrame(column_values_dict_rojo_overlap[idx]).transpose()
        df_r_ov.columns = image_names
        out_r_ov = os.path.join(
            summary_analysis_directory,
            f"{colname}_rojo_overlap_Track_statistics.csv"
        )
        df_r_ov.to_csv(out_r_ov, index=False)

    for idx in column_indexes_overlap_verde:
        colname = overlap_colnames_v.get(idx, f"col_{idx}")
        df_v_ov = pd.DataFrame(column_values_dict_verde_overlap[idx]).transpose()
        df_v_ov.columns = image_names
        out_v_ov = os.path.join(
            summary_analysis_directory,
            f"{colname}_verde_overlap_Track_statistics.csv"
        )
        df_v_ov.to_csv(out_v_ov, index=False)

    # ==========================================================
    # GUARDADOS: Versiones 1-columna con filas en bloque + línea en blanco
    # ==========================================================
    for idx in column_indexes_no_overlap:
        colname = no_overlap_colnames.get(idx, f"col_{idx}")

        # ROJO No overlap - 1 columna
        values_r = flatten_with_blanks(column_values_dict_rojo_no_overlap[idx])
        pd.DataFrame(values_r, columns=[colname]).to_csv(
            os.path.join(summary_analysis_directory, f"{colname}_rojo_No_overlap_Track_statistics_1column.csv"),
            index=False
        )

        # VERDE No overlap - 1 columna
        values_v = flatten_with_blanks(column_values_dict_verde_no_overlap[idx])
        pd.DataFrame(values_v, columns=[colname]).to_csv(
            os.path.join(summary_analysis_directory, f"{colname}_verde_No_overlap_Track_statistics_1column.csv"),
            index=False
        )

    for idx in column_indexes_overlap_rojo:
        colname = overlap_colnames_r.get(idx, f"col_{idx}")
        values_r = flatten_with_blanks(column_values_dict_rojo_overlap[idx])
        pd.DataFrame(values_r, columns=[colname]).to_csv(
            os.path.join(summary_analysis_directory, f"{colname}_rojo_overlap_Track_statistics_1column.csv"),
            index=False
        )

    for idx in column_indexes_overlap_verde:
        colname = overlap_colnames_v.get(idx, f"col_{idx}")
        values_v = flatten_with_blanks(column_values_dict_verde_overlap[idx])
        pd.DataFrame(values_v, columns=[colname]).to_csv(
            os.path.join(summary_analysis_directory, f"{colname}_verde_overlap_Track_statistics_1column.csv"),
            index=False
        )

    print("[OK] extract_column_values_single_file: resúmenes creados usando SOLO CSV filtrados por threshold.")


# Save each result to a separate CSV file with the image name before 'trajectory_overlap_results.csv'
summary_data = []
for image_name, file_results in results_spots.items():
    image_directory = os.path.join(results_directory, image_name)
    os.makedirs(image_directory, exist_ok=True)

    # Create a directory named "csv" in the image directory if it doesn't exist
    csv_directory = os.path.join(image_directory, "csv")
    os.makedirs(csv_directory, exist_ok=True)

    # Create a directory named "plots" in the image directory if it doesn't exist
    plots_directory = os.path.join(image_directory, "plots")
    os.makedirs(plots_directory, exist_ok=True)

    # Guardar archivo general
    results_df = pd.DataFrame(file_results)
    results_file_path = os.path.join(csv_directory, f'{image_name}_recalculated_trajectory_overlap_results.csv')
    results_df.to_csv(results_file_path, index=False)

    # Crear y guardar archivo solo con columnas 'Rojo' + 'Image Title'
    rojo_columns = [col for col in results_df.columns if 'Rojo' in col or col == 'Image Title']
    results_df_rojo = results_df[rojo_columns]
    results_file_path_rojo = os.path.join(csv_directory,
                                          f'{image_name}_rojo_recalculated_trajectory_overlap_results.csv')
    results_df_rojo.to_csv(results_file_path_rojo, index=False)

    # Crear y guardar archivo solo con columnas 'Verde' + 'Image Title'
    verde_columns = [col for col in results_df.columns if 'Verde' in col or col == 'Image Title']
    results_df_verde = results_df[verde_columns]
    results_file_path_verde = os.path.join(csv_directory,
                                           f'{image_name}_verde_recalculated_trajectory_overlap_results.csv')
    results_df_verde.to_csv(results_file_path_verde, index=False)

    # Collect summary data per image
    n_overlapping_tracks = len(file_results)
    n_red_tracks_overlapping = len(set([result['Rojo_TRACK_ID'] for result in file_results]))
    n_green_tracks_overlapping = len(set([result['Verde_TRACK_ID'] for result in file_results]))

    # Calculate non-overlapping tracks
    total_red_tracks = len(rojo_dfs_spots[image_name]['TRACK_ID'].unique())
    total_green_tracks = len(verde_dfs_spots[image_name]['TRACK_ID'].unique())
    n_red_tracks_no_overlapping = total_red_tracks - n_red_tracks_overlapping
    n_green_tracks_no_overlapping = total_green_tracks - n_green_tracks_overlapping
    n_no_overlapping_tracks = n_red_tracks_no_overlapping + n_green_tracks_no_overlapping

    summary_data.append({
        'Image Title': image_name,
        'N of Overlapping Tracks': n_overlapping_tracks,
        'N of Red Tracks overlapping': n_red_tracks_overlapping,
        'N of Green Tracks overlapping': n_green_tracks_overlapping,
        'N of No Overlapping Tracks': n_no_overlapping_tracks,
        'N of Red Tracks No Overlapping': n_red_tracks_no_overlapping,
        'N of Green Tracks No Overlapping': n_green_tracks_no_overlapping
    })

    # Generate plots for overlapping tracks
    rojo_df = rojo_dfs_spots[image_name]
    verde_df = verde_dfs_spots[image_name]
    # plot_overlapping_tracks(image_name, rojo_df, verde_df, file_results, plots_directory)

    for result in file_results:
        rojo_track = rojo_df[rojo_df['TRACK_ID'] == result['Rojo_TRACK_ID']]
        verde_track = verde_df[verde_df['TRACK_ID'] == result['Verde_TRACK_ID']]

        # Plot POSITION_X vs FRAME
        plt.figure()
        plt.plot(rojo_track['FRAME'], rojo_track['POSITION_X'], 'r-', label=f'Rojo {result["Rojo_TRACK_ID"]}')
        plt.plot(verde_track['FRAME'], verde_track['POSITION_X'], 'g-', label=f'Verde {result["Verde_TRACK_ID"]}')
        plt.xlabel('FRAME')
        plt.ylabel('POSITION_X')
        plt.ylim([rojo_track['POSITION_X'].min() - 2, verde_track['POSITION_X'].max() + 2])  # Adjust y-axis limits
        plt.title(f'{image_name} - POSITION_X vs FRAME')
        plt.legend()
        plt.savefig(os.path.join(plots_directory,
                                 f'{image_name}_POSITION_X_vs_FRAME_{result["Rojo_TRACK_ID"]}_{result["Verde_TRACK_ID"]}.png'))
        plt.close()

        # Plot POSITION_Y vs FRAME
        plt.figure()
        plt.plot(rojo_track['FRAME'], rojo_track['POSITION_Y'], 'r-', label=f'Rojo {result["Rojo_TRACK_ID"]}')
        plt.plot(verde_track['FRAME'], verde_track['POSITION_Y'], 'g-', label=f'Verde {result["Verde_TRACK_ID"]}')
        plt.xlabel('FRAME')
        plt.ylabel('POSITION_Y')
        plt.ylim([rojo_track['POSITION_Y'].min() - 2, verde_track['POSITION_Y'].max() + 2])  # Adjust y-axis limits
        plt.title(f'{image_name} - POSITION_Y vs FRAME')
        plt.legend()
        plt.savefig(os.path.join(plots_directory,
                                 f'{image_name}_POSITION_Y_vs_FRAME_{result["Rojo_TRACK_ID"]}_{result["Verde_TRACK_ID"]}.png'))
        plt.close()

        # # Create CSV file with POSITION_X, POSITION_Y, and FRAME
        # combined_df = pd.DataFrame({
        #     'Rojo_POSITION_X': rojo_track['POSITION_X'],
        #     'Rojo_POSITION_Y': rojo_track['POSITION_Y'],
        #     'Rojo_FRAME': rojo_track['FRAME'],
        #     'Verde_POSITION_X': verde_track['POSITION_X'],
        #     'Verde_POSITION_Y': verde_track['POSITION_Y'],
        #     'Verde_FRAME': verde_track['FRAME']
        # })
        # csv_filename = f'{image_name}_POSITION_XY_vs_FRAME_{result["Rojo_TRACK_ID"]}_{result["Verde_TRACK_ID"]}.csv'
        # combined_df.to_csv(os.path.join(plots_directory, csv_filename), index=False)

        # Perform a full outer join on FRAME to include all rows
        combined_df = pd.merge(rojo_track[['POSITION_X', 'POSITION_Y', 'FRAME']],
                               verde_track[['POSITION_X', 'POSITION_Y', 'FRAME']], on='FRAME', how='outer',
                               suffixes=('_Rojo', '_Verde'))

        # Create CSV file with POSITION_X, POSITION_Y, and FRAME
        csv_filename = f'{image_name}_POSITION_XY_vs_FRAME_{result["Rojo_TRACK_ID"]}_{result["Verde_TRACK_ID"]}.csv'
        combined_df.to_csv(os.path.join(plots_directory, csv_filename), index=False)

        # # Plot 3D scatter plot POSITION_X, POSITION_Y, FRAME
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(rojo_track['POSITION_X'], rojo_track['POSITION_Y'], rojo_track['FRAME'], 'r-', label='Rojo')
        # ax.plot(verde_track['POSITION_X'], verde_track['POSITION_Y'], verde_track['FRAME'], 'g-', label='Verde')
        # ax.set_xlabel('POSITION_X')
        # ax.set_ylabel('POSITION_Y')
        # ax.set_zlabel('FRAME')
        # ax.set_title(f'{image_name}_POSITION_XY_vs_FRAME')
        # plt.savefig(os.path.join(plots_directory,
        #                          f'{image_name}_POSITION_XY_vs_FRAME_{result["Rojo_TRACK_ID"]}_{result["Verde_TRACK_ID"]}.png'))
        # plt.close()

# Save summary data to a CSV file
summary_df = pd.DataFrame(summary_data)

# 1) Nueva columna: Total Tracks = Overlapping + No Overlapping
summary_df['Total Tracks'] = (
    summary_df['N of Overlapping Tracks'].fillna(0) +
    summary_df['N of No Overlapping Tracks'].fillna(0)
)

# 2) Fila final "Total" con el sumatorio de cada columna numérica
total_row = summary_df.drop(columns=['Image Title']).sum(numeric_only=True)
total_row['Image Title'] = 'Total'

# Mantener el mismo orden de columnas (Image Title primero, si lo está)
summary_df = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)

summary_df.to_csv(
    os.path.join(summary_analysis_directory, 'summary_trajectory_overlap_results.csv'),
    index=False
)


# Function to update the CSV spot_in tracks statistics files
def update_csv(df, file_name):
    # Calculate 'Intensity-Bg Rojo' and 'Intensity-Bg Verde'
    if 'rojo' in file_name:
        df['Intensity-Bg Rojo'] = df['MEAN_INTENSITY_CH1'] - df['Intensity-Bg Subtract']
    elif 'verde' in file_name:
        df['Intensity-Bg Verde'] = df['MEAN_INTENSITY_CH1'] - df['Intensity-Bg Subtract']

    # Rename 'Intensity-Bg Subtract' to 'Spot Intensity-Bg Subtract'
    df.rename(columns={'Intensity-Bg Subtract': 'Spot Intensity-Bg Subtract'}, inplace=True)
    # Remove the 'Unnamed: 21' column if it exists
    if 'Unnamed: 21' in df.columns:
        df.drop(columns=['Unnamed: 21'], inplace=True)
    return df


def calculate_average_values(df, track_id_column, columns_to_average):
    average_values = df.groupby(track_id_column)[columns_to_average].mean().reset_index()
    return average_values


# ==============================
# Process _Spots in tracks statistics.csv files (FILTRADO GLOBAL ANTES)
# ==============================

# Diccionarios para guardar promedios por imagen (los usaremos en el paso de Tracks)
avg_rojo_overlap_by_image = {}
avg_rojo_no_overlap_by_image = {}
avg_verde_overlap_by_image = {}
avg_verde_no_overlap_by_image = {}

for image_name in sorted(set(rojo_csv_files_spots) & set(verde_csv_files_spots)):

    # ====== FILTRO GLOBAL por IDs permitidos (threshold aplicado al inicio del pipeline) ======
    allowed_r = allowed_ids_rojo_by_image.get(image_name, set())
    allowed_v = allowed_ids_verde_by_image.get(image_name, set())

    # ✅ Si falta alguno de los dos colores, IGNORAR la imagen por completo
    if not allowed_r or not allowed_v:
        print(f"[SKIP] {image_name}: 0 tracks válidos en alguno de los colores (rojo={len(allowed_r)}, verde={len(allowed_v)})")
        continue

    # Cargar SPOTS de Rojo/Verde de forma segura
    rojo_spots_df = safe_read_csv(rojo_csv_files_spots[image_name], context=f"{image_name} rojo spots")
    verde_spots_df = safe_read_csv(verde_csv_files_spots[image_name], context=f"{image_name} verde spots")

    # ✅ Si algún CSV está vacío, IGNORAR la imagen
    if rojo_spots_df.empty or verde_spots_df.empty:
        print(f"[SKIP] {image_name}: spots CSV vacío (rojo_empty={rojo_spots_df.empty}, verde_empty={verde_spots_df.empty})")
        continue

    # Filtrar por IDs permitidos
    rojo_spots_df = rojo_spots_df[rojo_spots_df['TRACK_ID'].isin(allowed_r)].copy()
    verde_spots_df = verde_spots_df[verde_spots_df['TRACK_ID'].isin(allowed_v)].copy()

    # ✅ Si tras filtrar queda vacío algún color, IGNORAR la imagen
    if rojo_spots_df.empty or verde_spots_df.empty:
        print(f"[SKIP] {image_name}: sin spots tras filtrar por allowed IDs.")
        continue

    # IDs de overlap obtenidos del análisis de pares (si no hay, lista vacía)
    file_results = results_spots.get(image_name, [])
    overlapping_rojo_ids = set(r['Rojo_TRACK_ID'] for r in file_results)
    overlapping_verde_ids = set(r['Verde_TRACK_ID'] for r in file_results)

    # Separación overlap / no-overlap (dentro del conjunto ya filtrado)
    rojo_overlap_df = rojo_spots_df[rojo_spots_df['TRACK_ID'].isin(overlapping_rojo_ids)].copy()
    rojo_no_overlap_df = rojo_spots_df[~rojo_spots_df['TRACK_ID'].isin(overlapping_rojo_ids)].copy()
    verde_overlap_df = verde_spots_df[verde_spots_df['TRACK_ID'].isin(overlapping_verde_ids)].copy()
    verde_no_overlap_df = verde_spots_df[~verde_spots_df['TRACK_ID'].isin(overlapping_verde_ids)].copy()

    # Actualizar/renombrar columnas para coherencia
    rojo_overlap_df = update_csv(rojo_overlap_df, f'{image_name}_rojo_overlap_Spots_statistics.csv')
    rojo_no_overlap_df = update_csv(rojo_no_overlap_df, f'{image_name}_rojo_No_overlap_Spots_statistics.csv')
    verde_overlap_df = update_csv(verde_overlap_df, f'{image_name}_verde_overlap_Spots_statistics.csv')
    verde_no_overlap_df = update_csv(verde_no_overlap_df, f'{image_name}_verde_No_overlap_Spots_statistics.csv')

    # Calcular promedios por TRACK_ID (como en tu script original)
    columns_to_average_rojo = ['Spot Intensity-Bg Subtract', 'Intensity-Bg Rojo']
    columns_to_average_verde = ['Spot Intensity-Bg Subtract', 'Intensity-Bg Verde']

    rojo_avg_overlap = calculate_average_values(rojo_overlap_df, 'TRACK_ID', columns_to_average_rojo)
    rojo_avg_no_overlap = calculate_average_values(rojo_no_overlap_df, 'TRACK_ID', columns_to_average_rojo)
    verde_avg_overlap = calculate_average_values(verde_overlap_df, 'TRACK_ID', columns_to_average_verde)
    verde_avg_no_overlap = calculate_average_values(verde_no_overlap_df, 'TRACK_ID', columns_to_average_verde)

    # Guardar promedios por imagen
    avg_rojo_overlap_by_image[image_name] = rojo_avg_overlap
    avg_rojo_no_overlap_by_image[image_name] = rojo_avg_no_overlap
    avg_verde_overlap_by_image[image_name] = verde_avg_overlap
    avg_verde_no_overlap_by_image[image_name] = verde_avg_no_overlap

    # Crear directorios de salida por imagen SOLO si decidimos procesarla
    image_directory = os.path.join(results_directory, image_name)
    os.makedirs(image_directory, exist_ok=True)
    csv_directory = os.path.join(image_directory, "csv")
    os.makedirs(csv_directory, exist_ok=True)
    plots_directory = os.path.join(image_directory, "plots")
    os.makedirs(plots_directory, exist_ok=True)

    # Guardar los SPOTS (ya filtrados y separados)
    rojo_overlap_df.to_csv(os.path.join(csv_directory, f'{image_name}_rojo_overlap_Spots_statistics.csv'), index=False)
    rojo_no_overlap_df.to_csv(os.path.join(csv_directory, f'{image_name}_rojo_No_overlap_Spots_statistics.csv'), index=False)
    verde_overlap_df.to_csv(os.path.join(csv_directory, f'{image_name}_verde_overlap_Spots_statistics.csv'), index=False)
    verde_no_overlap_df.to_csv(os.path.join(csv_directory, f'{image_name}_verde_No_overlap_Spots_statistics.csv'), index=False)

    # Plots solo si hay pares overlap
    if file_results:
        plot_overlapping_tracks(image_name, rojo_overlap_df, verde_overlap_df, file_results, plots_directory)
    else:
        print(f"[INFO] {image_name}: sin pares overlap; se guardaron únicamente los CSV de No_overlap (filtrados).")

# ==============================
# Process _Tracks statistics.csv files (FILTRADO GLOBAL ANTES)
# ==============================

for image_name in sorted(rojo_csv_files_tracks.keys()):

    if image_name not in verde_csv_files_tracks:
        continue

    # ✅ Recuperar IDs permitidos ANTES de leer CSV
    allowed_r = allowed_ids_rojo_by_image.get(image_name, set())
    allowed_v = allowed_ids_verde_by_image.get(image_name, set())

    # ✅ Si falta alguno de los dos colores, IGNORAR la imagen
    if not allowed_r or not allowed_v:
        print(f"[SKIP] {image_name}: 0 tracks válidos en alguno de los colores (rojo={len(allowed_r)}, verde={len(allowed_v)})")
        continue

    # ✅ Leer CSV de Tracks de forma segura (evita EmptyDataError)
    rojo_tracks_df = safe_read_csv(rojo_csv_files_tracks[image_name], context=f"{image_name} rojo Tracks")
    verde_tracks_df = safe_read_csv(verde_csv_files_tracks[image_name], context=f"{image_name} verde Tracks")

    # ✅ Si alguno está vacío (archivo vacío o sin columnas), IGNORAR la imagen
    if rojo_tracks_df.empty or verde_tracks_df.empty:
        print(f"[SKIP] {image_name}: Tracks statistics vacío en alguno de los colores "
              f"(rojo_empty={rojo_tracks_df.empty}, verde_empty={verde_tracks_df.empty})")
        continue

    # Filtrar por IDs permitidos
    rojo_tracks_df = rojo_tracks_df[rojo_tracks_df['TRACK_ID'].isin(allowed_r)].copy()
    verde_tracks_df = verde_tracks_df[verde_tracks_df['TRACK_ID'].isin(allowed_v)].copy()

    # ✅ Si tras filtrar queda vacío algún color, IGNORAR la imagen
    if rojo_tracks_df.empty or verde_tracks_df.empty:
        print(f"[SKIP] {image_name}: sin tracks tras filtrar por allowed IDs.")
        continue

    # IDs de overlap
    file_results = results_spots.get(image_name, [])
    overlapping_rojo_ids = set(r['Rojo_TRACK_ID'] for r in file_results)
    overlapping_verde_ids = set(r['Verde_TRACK_ID'] for r in file_results)

    # Separación overlap / no-overlap
    rojo_overlap_tracks_df = rojo_tracks_df[rojo_tracks_df['TRACK_ID'].isin(overlapping_rojo_ids)].copy()
    rojo_no_overlap_tracks_df = rojo_tracks_df[~rojo_tracks_df['TRACK_ID'].isin(overlapping_rojo_ids)].copy()
    verde_overlap_tracks_df = verde_tracks_df[verde_tracks_df['TRACK_ID'].isin(overlapping_verde_ids)].copy()
    verde_no_overlap_tracks_df = verde_tracks_df[~verde_tracks_df['TRACK_ID'].isin(overlapping_verde_ids)].copy()

    # Recuperar promedios calculados en el paso de Spots
    def safe_avg(df_like):
        return df_like if df_like is not None else pd.DataFrame(columns=['TRACK_ID'])

    rojo_avg_overlap = safe_avg(avg_rojo_overlap_by_image.get(image_name))
    rojo_avg_no_overlap = safe_avg(avg_rojo_no_overlap_by_image.get(image_name))
    verde_avg_overlap = safe_avg(avg_verde_overlap_by_image.get(image_name))
    verde_avg_no_overlap = safe_avg(avg_verde_no_overlap_by_image.get(image_name))

    # Merge de promedios (left join por TRACK_ID)
    rojo_overlap_tracks_df = rojo_overlap_tracks_df.merge(rojo_avg_overlap, on='TRACK_ID', how='left')
    rojo_no_overlap_tracks_df = rojo_no_overlap_tracks_df.merge(rojo_avg_no_overlap, on='TRACK_ID', how='left')
    verde_overlap_tracks_df = verde_overlap_tracks_df.merge(verde_avg_overlap, on='TRACK_ID', how='left')
    verde_no_overlap_tracks_df = verde_no_overlap_tracks_df.merge(verde_avg_no_overlap, on='TRACK_ID', how='left')

    # Crear directorios SOLO si la imagen se procesa
    image_directory = os.path.join(results_directory, image_name)
    os.makedirs(image_directory, exist_ok=True)
    csv_directory = os.path.join(image_directory, "csv")
    os.makedirs(csv_directory, exist_ok=True)

    # Guardar los TRACKS actualizados
    rojo_overlap_tracks_df.to_csv(os.path.join(csv_directory, f'{image_name}_rojo_overlap_Track_statistics.csv'), index=False)
    rojo_no_overlap_tracks_df.to_csv(os.path.join(csv_directory, f'{image_name}_rojo_No_overlap_Track_statistics.csv'), index=False)
    verde_overlap_tracks_df.to_csv(os.path.join(csv_directory, f'{image_name}_verde_overlap_Track_statistics.csv'), index=False)
    verde_no_overlap_tracks_df.to_csv(os.path.join(csv_directory, f'{image_name}_verde_No_overlap_Track_statistics.csv'), index=False)

    print(f"[OK] {image_name}: Tracks guardados -> "
          f"Rojo(overlap={len(rojo_overlap_tracks_df)}, no_overlap={len(rojo_no_overlap_tracks_df)}), "
          f"Verde(overlap={len(verde_overlap_tracks_df)}, no_overlap={len(verde_no_overlap_tracks_df)})")

# Extract specified column values and save to CSV
extract_column_values_single_file(column_indexes_no_overlap, column_indexes_overlap_rojo, column_indexes_overlap_verde)

import os
import pandas as pd


def create_summary_csv(results_spots, rojo_dfs_spots, verde_dfs_spots, summary_analysis_directory):
    summary_data_rojo = []
    summary_data_verde = []

    for image_name, file_results in results_spots.items():
        results_file_path = os.path.join(
            results_directory,  # <- asumido como variable global
            image_name, "csv",
            f"{image_name}_recalculated_trajectory_overlap_results.csv"
        )
        results_df = pd.read_csv(results_file_path)
        total_tracks = len(results_df)
        n_overlapping_tracks = len(file_results)
        n_red_tracks_overlapping = len(set([result['Rojo_TRACK_ID'] for result in file_results]))
        n_green_tracks_overlapping = len(set([result['Verde_TRACK_ID'] for result in file_results]))

        # NUEVAS CLAVES
        base_counts = {
            'Immobile Tracks': 0, 'Short Tracks': 0, 'Long Tracks': 0,
            'Short Inmobile': 0, 'Short Mobile': 0,
            'Short Mobile Confined': 0, 'Short Mobile Anomalous': 0,
            'Short Mobile Free': 0, 'Short Mobile Directed': 0,
            'Long Mobile': 0, 'Long Confined': 0, 'Long Free': 0,
            'Long Directed': 0, 'Long Mobile Confined': 0,
            'Long Mobile Free': 0, 'Long Mobile Directed': 0
        }
        rojo_classification_counts = base_counts.copy()
        verde_classification_counts = base_counts.copy()

        # Helpers para incrementar evitando KeyError
        def inc(d, key):
            if key in d:
                d[key] += 1

        for _, row in results_df.iterrows():
            # =========================
            #        ROJO
            # =========================
            # Track length
            if row['Track Length Rojo'] == 'Short':
                inc(rojo_classification_counts, 'Short Tracks')

                if row['Motility Rojo'] == 'Inmobile':
                    inc(rojo_classification_counts, 'Short Inmobile')
                    inc(rojo_classification_counts, 'Immobile Tracks')
                elif row['Motility Rojo'] == 'Mobile':
                    inc(rojo_classification_counts, 'Short Mobile')
                    short_move = str(row['Alpha Movement Rojo'])
                    inc(rojo_classification_counts, f"Short Mobile {short_move}")
                # Si hubiera otros valores de motilidad, se ignoran
            else:
                # Long
                inc(rojo_classification_counts, 'Long Tracks')

                long_move = str(row['sMSS Rojo Movement'])
                inc(rojo_classification_counts, f"Long {long_move}")  # Long Confined/Free/Directed

                if row['Motility Rojo'] == 'Mobile':
                    inc(rojo_classification_counts, 'Long Mobile')
                    inc(rojo_classification_counts, f"Long Mobile {long_move}")
                elif row['Motility Rojo'] == 'Inmobile':
                    inc(rojo_classification_counts, 'Immobile Tracks')

            # =========================
            #        VERDE
            # =========================
            if row['Track Length Verde'] == 'Short':
                inc(verde_classification_counts, 'Short Tracks')

                if row['Motility Verde'] == 'Inmobile':
                    inc(verde_classification_counts, 'Short Inmobile')
                    inc(verde_classification_counts, 'Immobile Tracks')
                elif row['Motility Verde'] == 'Mobile':
                    inc(verde_classification_counts, 'Short Mobile')
                    short_move_v = str(row['Alpha Movement Verde'])
                    inc(verde_classification_counts, f"Short Mobile {short_move_v}")
            else:
                inc(verde_classification_counts, 'Long Tracks')

                long_move_v = str(row['sMSS Verde Movement'])
                inc(verde_classification_counts, f"Long {long_move_v}")

                if row['Motility Verde'] == 'Mobile':
                    inc(verde_classification_counts, 'Long Mobile')
                    inc(verde_classification_counts, f"Long Mobile {long_move_v}")
                elif row['Motility Verde'] == 'Inmobile':
                    inc(verde_classification_counts, 'Immobile Tracks')

        summary_data_rojo.append({
            'Image Title': image_name,
            'Total Tracks': total_tracks,
            'N of Overlapping Tracks': n_overlapping_tracks,
            'N of Red Tracks overlapping': n_red_tracks_overlapping,
            **rojo_classification_counts
        })

        summary_data_verde.append({
            'Image Title': image_name,
            'Total Tracks': total_tracks,
            'N of Overlapping Tracks': n_overlapping_tracks,
            'N of Green Tracks overlapping': n_green_tracks_overlapping,
            **verde_classification_counts
        })

    # Crear DataFrames
    summary_df_rojo = pd.DataFrame(summary_data_rojo)
    summary_df_verde = pd.DataFrame(summary_data_verde)

    # Añadir fila 'Total' al final
    total_row_rojo = summary_df_rojo.drop(columns=['Image Title']).sum(numeric_only=True)
    total_row_rojo['Image Title'] = 'Total'
    summary_df_rojo = pd.concat([summary_df_rojo, pd.DataFrame([total_row_rojo])], ignore_index=True)

    total_row_verde = summary_df_verde.drop(columns=['Image Title']).sum(numeric_only=True)
    total_row_verde['Image Title'] = 'Total'
    summary_df_verde = pd.concat([summary_df_verde, pd.DataFrame([total_row_verde])], ignore_index=True)

    # Guardar los archivos
    summary_df_rojo.to_csv(os.path.join(summary_analysis_directory, 'summary_track_condition_rojo_overlapping.csv'),
                           index=False)
    summary_df_verde.to_csv(os.path.join(summary_analysis_directory, 'summary_track_condition_verde_overlapping.csv'),
                            index=False)


import numpy as np
import pandas as pd
import os


def create_summary_csv_no_overlap(results_spots, rojo_dfs_spots, verde_dfs_spots, summary_analysis_directory):
    """
    Genera dos CSV de resumen para tracks NO overlapping:
      - summary_track_condition_rojo_no_overlapping.csv
      - summary_track_condition_verde_no_overlapping.csv

    Clasificaciones y umbrales coherentes con analyze_trajectories:
      * Track Length:  'Long' si len(track) >= length_threshold; si no, 'Short'
      * Motility:      'Mobile' si D1-4 >= motility_threshold; si no, 'Inmobile'
      * Alpha movement (para Short Mobile): Confined / Anomalous / Free / Directed
      * sMSS movement  (para Long):         Confined / Free / Directed
    """

    # Diccionario base de contadores (idéntico a tu create_summary_csv de overlapping)
    base_counts = {
        'Immobile Tracks': 0, 'Short Tracks': 0, 'Long Tracks': 0,
        'Short Inmobile': 0, 'Short Mobile': 0,
        'Short Mobile Confined': 0, 'Short Mobile Anomalous': 0,
        'Short Mobile Free': 0, 'Short Mobile Directed': 0,
        'Long Mobile': 0, 'Long Confined': 0, 'Long Free': 0,
        'Long Directed': 0, 'Long Mobile Confined': 0,
        'Long Mobile Free': 0, 'Long Mobile Directed': 0
    }

    def inc(d, key):
        if key in d:
            d[key] += 1

    def classify_alpha_movement(alpha_val):
        # Igual que en analyze_trajectories
        if alpha_val is None or np.isnan(alpha_val):
            return 'Undefined'
        if 0 <= alpha_val < 0.6:
            return 'Confined'
        elif 0.6 <= alpha_val < 0.9:
            return 'Anomalous'
        elif 0.9 <= alpha_val < 1.1:
            return 'Free'
        elif alpha_val >= 1.1:
            return 'Directed'
        else:
            return 'Undefined'

    def classify_smss_movement(smss_val):
        # Igual que en analyze_trajectories
        if smss_val is None or np.isnan(smss_val):
            return 'Undefined'
        if smss_val == 1.0:
            return 'Unidirectional Ballistic'  # se ignorará en contadores (no existe clave)
        elif smss_val == 0:
            return 'Immobility'  # se ignorará en contadores (no existe clave)
        elif 0.45 <= smss_val <= 0.55:
            return 'Free'
        elif 0 < smss_val < 0.45:
            return 'Confined'
        elif smss_val > 0.55:
            return 'Directed'
        else:
            return 'Undefined'

    def safe_metrics_from_track(track_xy):
        """
        Calcula MSD, Alpha, sMSS y D-slopes de forma robusta para trayectorias cortas.
        Devuelve: (alpha, smss, D_slope_1lag, D_slope_1to4)
        """
        n = len(track_xy)
        if n < 2:
            # Sin desplazamiento posible; clasificar como inmóvil
            return (np.nan, 0.0, 0.0, 0.0)

        # Usamos hasta 4 lags pero no más de n-1
        max_lag = min(4, n - 1)
        time_lags = np.arange(1, max_lag + 1)
        msd = calculate_msd(track_xy, max_lag)

        # Alpha (log-log fit) – requiere al menos 2 puntos
        try:
            if len(time_lags) >= 2:
                alpha_val = abs(calculate_alpha(msd, time_lags))
            else:
                alpha_val = np.nan
        except Exception:
            alpha_val = np.nan

        # sMSS
        try:
            smss_val = abs(calculate_sMSS(track_xy, max_lag))
        except Exception:
            smss_val = np.nan

        # D slopes (mantenemos tu enfoque: fit lineal aunque haya 1 punto; numpy devuelve coef con RankWarning)
        try:
            D_slope_1lag, _ = abs(np.polyfit(time_lags[:1], msd[:1], 1))
        except Exception:
            D_slope_1lag = 0.0

        try:
            upto = min(4, len(time_lags))
            # si solo hay 1 punto, polyfit(1 punto, deg=1) devuelve coef con RankWarning; capturamos por si acaso
            D_slope_1to4, _ = abs(np.polyfit(time_lags[:upto], msd[:upto], 1))
        except Exception:
            D_slope_1to4 = 0.0

        return (alpha_val, smss_val, D_slope_1lag, D_slope_1to4)

    def summarize_color_no_overlap(image_name, df_spots, overlapping_ids_set, color_label):
        """
        Devuelve (summary_row_dict, non_overlapping_ids_set) para un color en una imagen.
        color_label: 'Rojo' o 'Verde'
        """
        counts = base_counts.copy()

        # IDs no overlapping para este color
        all_ids = set(df_spots['TRACK_ID'].unique())
        non_overlap_ids = all_ids - overlapping_ids_set

        # Recorremos cada track no overlapping
        for track_id, track_df in df_spots[df_spots['TRACK_ID'].isin(non_overlap_ids)].groupby('TRACK_ID'):
            traj = track_df[['POSITION_X', 'POSITION_Y']].values
            n = len(traj)

            # Long/Short
            length_cat = 'Long' if n >= length_threshold else 'Short'

            # Métricas
            alpha_val, smss_val, _, D1_4 = safe_metrics_from_track(traj)

            # Motilidad por umbral D1-4
            motility = 'Mobile' if D1_4 >= motility_threshold else 'Inmobile'

            # Clasificaciones por alpha / sMSS
            alpha_mov = classify_alpha_movement(alpha_val)
            smss_mov = classify_smss_movement(smss_val)

            # Contadores (mismas reglas que en tu resumen de overlapping)
            if length_cat == 'Short':
                inc(counts, 'Short Tracks')
                if motility == 'Inmobile':
                    inc(counts, 'Short Inmobile')
                    inc(counts, 'Immobile Tracks')
                elif motility == 'Mobile':
                    inc(counts, 'Short Mobile')
                    inc(counts,
                        f"Short Mobile {alpha_mov}")  # solo suma si clave existe (Confined/Anomalous/Free/Directed)

            else:  # Long
                inc(counts, 'Long Tracks')
                inc(counts, f"Long {smss_mov}")  # solo Confined/Free/Directed suman

                if motility == 'Mobile':
                    inc(counts, 'Long Mobile')
                    inc(counts, f"Long Mobile {smss_mov}")  # solo Confined/Free/Directed suman
                elif motility == 'Inmobile':
                    inc(counts, 'Immobile Tracks')

        # Fila resumen por imagen para este color
        summary_row = {
            'Image Title': image_name,
            # Total Tracks = nº de tracks no overlapping de este color
            'Total Tracks': len(non_overlap_ids),
            # Incluimos ambos contadores de no-overlap (por simetría con tu resumen de overlapping)
            'N of No Overlapping Tracks': len(non_overlap_ids),  # (para el color actual)
        }
        summary_row.update(counts)
        return summary_row, non_overlap_ids

    # ============================
    # Construimos los resúmenes
    # ============================
    summary_data_rojo = []
    summary_data_verde = []

    for image_name, file_results in results_spots.items():
        # Conjuntos de IDs overlapping por color en esta imagen
        overlapping_rojo_ids = {r['Rojo_TRACK_ID'] for r in file_results}
        overlapping_verde_ids = {r['Verde_TRACK_ID'] for r in file_results}

        # DataFrames completos de spots (por imagen)
        rojo_df = rojo_dfs_spots[image_name]
        verde_df = verde_dfs_spots[image_name]

        # ROJO
        row_rojo, nonover_rojo = summarize_color_no_overlap(
            image_name, rojo_df, overlapping_rojo_ids, color_label='Rojo'
        )
        summary_data_rojo.append(row_rojo)

        # VERDE
        row_verde, nonover_verde = summarize_color_no_overlap(
            image_name, verde_df, overlapping_verde_ids, color_label='Verde'
        )
        summary_data_verde.append(row_verde)

    # DataFrames finales + fila Total
    summary_df_rojo = pd.DataFrame(summary_data_rojo)
    summary_df_verde = pd.DataFrame(summary_data_verde)

    total_row_rojo = summary_df_rojo.drop(columns=['Image Title']).sum(numeric_only=True)
    total_row_rojo['Image Title'] = 'Total'
    summary_df_rojo = pd.concat([summary_df_rojo, pd.DataFrame([total_row_rojo])], ignore_index=True)

    total_row_verde = summary_df_verde.drop(columns=['Image Title']).sum(numeric_only=True)
    total_row_verde['Image Title'] = 'Total'
    summary_df_verde = pd.concat([summary_df_verde, pd.DataFrame([total_row_verde])], ignore_index=True)

    # Guardado
    summary_df_rojo.to_csv(
        os.path.join(summary_analysis_directory, 'summary_track_condition_rojo_no_overlapping.csv'),
        index=False
    )
    summary_df_verde.to_csv(
        os.path.join(summary_analysis_directory, 'summary_track_condition_verde_no_overlapping.csv'),
        index=False
    )
def update_total_tracks_summary_track_condition(summary_analysis_directory):
    """
    Ajusta 'Total Tracks' en los CSV que empiezan por 'summary_track_condition'
    para que sea: N of Overlapping Tracks + N of No Overlapping Tracks.

    Como esas columnas están separadas en:
      - *_overlapping.csv
      - *_no_overlapping.csv
    se leen ambos, se unen por 'Image Title' y se reescriben con Total Tracks corregido.
    También rehace la fila final 'Total' (sumatorio de columnas numéricas).
    """
    import os
    import pandas as pd
    import numpy as np

    def _read_no_total(path):
        df = pd.read_csv(path)
        # Eliminar fila "Total" si existe para no duplicar/contaminar el merge
        if 'Image Title' in df.columns:
            df = df[df['Image Title'] != 'Total'].copy()
        return df

    def _readd_total_row(df):
        # Recalcular la fila "Total" como suma de columnas numéricas
        total_vals = df.drop(columns=['Image Title'], errors='ignore').sum(numeric_only=True)

        # Crear fila total con TODAS las columnas (mismo orden), rellenando lo que no sea numérico con vacío
        total_row = {c: "" for c in df.columns}
        total_row['Image Title'] = 'Total'
        for k, v in total_vals.items():
            if k in total_row:
                total_row[k] = v

        return pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)

    def _move_total_tracks_to_col1(df):
        """
        Reordena columnas para que:
        0: Image Title
        1: Total Tracks
        y el resto después en el orden que ya tengan.
        """
        cols = list(df.columns)
        if 'Image Title' in cols and 'Total Tracks' in cols:
            new_cols = ['Image Title', 'Total Tracks'] + [c for c in cols if c not in ('Image Title', 'Total Tracks')]
            df = df[new_cols]
        return df

    for color in ["rojo", "verde"]:
        overlap_path = os.path.join(
            summary_analysis_directory, f"summary_track_condition_{color}_overlapping.csv"
        )
        no_overlap_path = os.path.join(
            summary_analysis_directory, f"summary_track_condition_{color}_no_overlapping.csv"
        )

        if not (os.path.exists(overlap_path) and os.path.exists(no_overlap_path)):
            print(f"[WARN] No encuentro ambos ficheros para '{color}'. Se omite ajuste de Total Tracks.")
            continue

        df_ov = _read_no_total(overlap_path)
        df_no = _read_no_total(no_overlap_path)

        # Asegurar que las columnas necesarias existen
        if 'N of Overlapping Tracks' not in df_ov.columns:
            raise KeyError(f"En {overlap_path} falta 'N of Overlapping Tracks'")
        if 'N of No Overlapping Tracks' not in df_no.columns:
            raise KeyError(f"En {no_overlap_path} falta 'N of No Overlapping Tracks'")

        # Merge por Image Title para poder sumar
        merged = df_ov[['Image Title', 'N of Overlapping Tracks']].merge(
            df_no[['Image Title', 'N of No Overlapping Tracks']],
            on='Image Title',
            how='outer'
        )

        # NaN -> 0 para poder sumar con seguridad
        merged['N of Overlapping Tracks'] = merged['N of Overlapping Tracks'].fillna(0)
        merged['N of No Overlapping Tracks'] = merged['N of No Overlapping Tracks'].fillna(0)

        merged['Total Tracks'] = merged['N of Overlapping Tracks'] + merged['N of No Overlapping Tracks']

        # Aplicar el Total Tracks corregido a cada df
        df_ov = df_ov.merge(merged[['Image Title', 'Total Tracks']], on='Image Title', how='left')
        df_no = df_no.merge(merged[['Image Title', 'Total Tracks']], on='Image Title', how='left')

        # (Opcional) si ya existía Total Tracks, aseguramos que quede con el valor nuevo
        # y no se duplique por conflictos de merge:
        # Si hubiera columnas Total Tracks_x/Total Tracks_y por merges previos, normalizamos.
        for df in (df_ov, df_no):
            cols = df.columns.tolist()
            if 'Total Tracks_x' in cols and 'Total Tracks_y' in cols:
                df.drop(columns=['Total Tracks_x'], inplace=True)
                df.rename(columns={'Total Tracks_y': 'Total Tracks'}, inplace=True)


        # Colocar Total Tracks como segunda columna (índice 1)
        df_ov = _move_total_tracks_to_col1(df_ov)
        df_no = _move_total_tracks_to_col1(df_no)


        # Reinsertar fila "Total" recalculada
        df_ov = _readd_total_row(df_ov)
        df_no = _readd_total_row(df_no)

        # Guardar de vuelta
        df_ov.to_csv(overlap_path, index=False)
        df_no.to_csv(no_overlap_path, index=False)

        print(f"[OK] Total Tracks actualizado en summary_track_condition_{color}_(overlapping/no_overlapping).csv")

# Ejecutar al final del script
create_summary_csv(results_spots, rojo_dfs_spots, verde_dfs_spots, summary_analysis_directory)
create_summary_csv_no_overlap(results_spots, rojo_dfs_spots, verde_dfs_spots, summary_analysis_directory)
# Ajustar Total Tracks = Overlapping + No Overlapping en summary_track_condition_*
update_total_tracks_summary_track_condition(summary_analysis_directory)