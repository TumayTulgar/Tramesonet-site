# =============================================================================
# TRAMESONET - OTONOM KİNEMATİK & ADVEKSİYON AJANI (GITHUB ACTIONS)
# ECMWF Grib + Spatial DEM + Navier Stokes Fiziği
# Telegram vb. yan modüllerden arındırılmış, saf veri toplama versiyonudur.
# =============================================================================

import os
import sys
import math
import json
import requests
import warnings
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import xarray as xr
import rasterio
from scipy.interpolate import Rbf

# MetPy ve Matplotlib (Headless)
import matplotlib
matplotlib.use('Agg')
import metpy.calc as mpcalc
from metpy.units import units

warnings.filterwarnings("ignore")

# =============================================================================
# GÜVENLİK VE ÇEVRE DEĞİŞKENLERİ (GITHUB SECRETS)
# =============================================================================
W_KEY = os.environ.get("WUNDERGROUND_API_KEY", "SECRET_YOK")
FIREBASE_URL = "https://tramesonet-wunderground-default-rtdb.firebaseio.com/"
FIREBASE_SECRET = os.environ.get("FIREBASE_SECRET", "")

# Fiziksel Sabitler
G = 9.80665 * units('m/s^2')
MU = 0.1 / units('s')
DT = 1.0 * units('s')
PBL_DEPTH = 1000.0 * units.m
PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

STATIONS = [
    "IKARTA22", "IISTAN68", "ISTANBUL189", "IATAEH6", "IATAEH5", "IMRANI1", 
    "IISTAN54", "ISKDAR4", "I34RESAD2", "IEKMEKY2", "IISTAN71", "IBEYKO4", 
    "ISARYER4", "ISARIY12", "ISARIY7", "ISARIY6", "IBEIKT3", "IBEIKT10", 
    "IISTAN85", "ISTANBUL182", "IBAHEL4", "IISTAN32", "IISTAN52", "IBAAKE10", 
    "IEYP6", "IISTAN67", "IBABAE3", "IERKEZ2", "IHAYRA1", "IISTAN24", 
    "IISTAN26", "IISTAN38", "IISTAN83", "IISTANBU108", "IKAPAKLI2", 
    "IKEAN12", "ILALAP1", "ISILIV12", "ISILIV8", "ITEKIR9", "ITEKIRDA7", 
    "IUZUNK2", "ISARAY2", "IATALCA4", "IKIRKL16", "IALPUL2", "IKEAN10"
]

# =============================================================================
# YARDIMCI VE I/O FONKSİYONLARI
# =============================================================================
def send_to_firebase(data, timestamp_str):
    path = f"{FIREBASE_URL}/logs/{timestamp_str}/{data['Station']}.json"
    if FIREBASE_SECRET: path += f"?auth={FIREBASE_SECRET}"
    try:
        res = requests.put(path, data=json.dumps(data), timeout=10)
        if res.status_code != 200: print(f" [!] {data['Station']} DB Hatası: {res.text}")
    except Exception as e: print(f" [!] Bağlantı Hatası: {e}")

class DEMLoader:
    def __init__(self, tif_path="output_hh.tif"):
        self.dataset = rasterio.open(tif_path) if os.path.exists(tif_path) else None
    def get_elevation_and_gradient(self, lat, lon, step_m=500):
        if not self.dataset: return 0.0, 0.0, 0.0
        try:
            row, col = self.dataset.index(lon, lat)
            elev = float(self.dataset.read(1)[row, col])
            dlat = step_m / 111000.0
            dlon = step_m / (111000.0 * math.cos(math.radians(lat)))
            h_e = float(self.dataset.read(1)[self.dataset.index(lon + dlon, lat)])
            h_w = float(self.dataset.read(1)[self.dataset.index(lon - dlon, lat)])
            h_n = float(self.dataset.read(1)[self.dataset.index(lon, lat + dlat)])
            h_s = float(self.dataset.read(1)[self.dataset.index(lon, lat - dlat)])
            return elev, (h_e - h_w)/(2*step_m), (h_n - h_s)/(2*step_m)
        except: return 0.0, 0.0, 0.0

# =============================================================================
# KİNEMATİK VE TRAJEKTÖR FİZİĞİ
# =============================================================================
def calculate_regional_kinematics(station_data_list):
    valid = [d for d in station_data_list if not np.isnan(d['u']) and not np.isnan(d['v'])]
    if len(valid) < 4:
        for d in station_data_list: d['divergence'] = 0.0
        return station_data_list
    lats, lons = np.array([d['lat'] for d in valid]), np.array([d['lon'] for d in valid])
    u_w, v_w = np.array([d['u'] for d in valid]), np.array([d['v'] for d in valid])
    m_lat, m_lon = lats.mean(), lons.mean()
    x_m = (lons - m_lon) * 111000 * math.cos(math.radians(m_lat))
    y_m = (lats - m_lat) * 111000
    r_u = Rbf(x_m, y_m, u_w, function='gaussian', epsilon=15000, smooth=0.1)
    r_v = Rbf(x_m, y_m, v_w, function='gaussian', epsilon=15000, smooth=0.1)
    for s in station_data_list:
        try:
            sx = (s['lon'] - m_lon) * 111000 * math.cos(math.radians(m_lat))
            sy = (s['lat'] - m_lat) * 111000
            dx = 500.0
            s['divergence'] = ((r_u(sx+dx, sy)-r_u(sx-dx, sy))/(2*dx)) + ((r_v(sx, sy+dx)-r_v(sx, sy-dx))/(2*dx))
        except: s['divergence'] = 0.0
    return station_data_list

class KineticParcel:
    def __init__(self, start_w=0.0):
        self.w = start_w * units('m/s')
        self.drift_x, self.drift_y, self.total_time = 0.0, 0.0, 0.0

    def solve(self, p, t, td, u_prof, v_prof, indices):
        path = []
        parcel_t_prof = indices['Profile']
        for i in range(len(p) - 1):
            tv_e = mpcalc.virtual_temperature(t[i], mpcalc.saturation_mixing_ratio(p[i], td[i]))
            tv_p = mpcalc.virtual_temperature(parcel_t_prof[i], mpcalc.saturation_mixing_ratio(p[i], td[0]))
            dw_dt = G * ((tv_p - tv_e) / tv_e) - (MU * self.w)
            self.w += dw_dt * DT
            self.drift_x += u_prof[i].magnitude * DT.magnitude
            self.drift_y += v_prof[i].magnitude * DT.magnitude
            self.total_time += DT.magnitude
            if self.w < 0 * units('m/s'): break
            path.append({'p': p[i].magnitude, 'w': self.w.magnitude})
        
        d_dist = math.sqrt(self.drift_x**2 + self.drift_y**2)
        d_ang = (270 - math.degrees(math.atan2(self.drift_y, self.drift_x))) % 360
        return pd.DataFrame(path), {"drift_x_m": round(self.drift_x, 1), "drift_y_m": round(self.drift_y, 1), "total_drift_m": round(d_dist, 1), "drift_direction": round(d_ang, 1), "ascent_time_s": self.total_time}

# =============================================================================
# ANA OPERASYON
# =============================================================================
if __name__ == "__main__":
    print("=== TRAMESONET OTONOM AJANI BAŞLATILDI ===")
    
    if W_KEY == "SECRET_YOK":
        print("[!] HATA: WUNDERGROUND_API_KEY bulunamadı. Çıkılıyor.")
        sys.exit(1)

    dem = DEMLoader("output_hh.tif")
    
    print("-> ECMWF Grib Verisi İndiriliyor...")
    from ecmwf.opendata import Client as ECMWFClient
    ecmwf_client = ECMWFClient(source="ecmwf")
    try:
        ecmwf_client.retrieve(type="fc", step=0, param=["t", "r", "u", "v"], levelist=PRESSURE_LEVELS, target="ecmwf_data.grib")
    except Exception as e:
        print(f"[!] ECMWF Hatası: {e}")
        sys.exit(1)

    print("-> İstasyon Verileri Çekiliyor...")
    raw_data = []
    for s_id in STATIONS:
        try:
            url = f"https://api.weather.com/v2/pws/observations/current?stationId={s_id}&format=json&units=m&apiKey={W_KEY}"
            d = requests.get(url, timeout=5).json()['observations'][0]
            m = d['metric']
            ws = (m['windSpeed']*1000/3600)*units('m/s')
            u, v = mpcalc.wind_components(ws, d['winddir']*units.degrees)
            raw_data.append({
                "id": s_id, "lat": d['lat'], "lon": d['lon'], "t": m['temp'], 
                "td": m.get('dewpt') or m['temp']-2, "u": u.magnitude, "v": v.magnitude, 
                "speed": m['windSpeed'], "dir": d['winddir'], "precip_total": m.get('precipTotal', 0)
            })
        except: pass

    network = calculate_regional_kinematics(raw_data)
    safe_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"-> Analiz ve Firebase Senkronizasyonu Başlıyor ({safe_timestamp})")

    ds = xr.open_dataset("ecmwf_data.grib", engine="cfgrib", backend_kwargs={'indexpath':''})

    for s in network:
        try:
            prof = ds.sel(latitude=s['lat'], longitude=s['lon'], method="nearest")
            p, t = prof.isobaricInhPa.values * units.hPa, (prof.t.values-273.15) * units.degC
            td = mpcalc.dewpoint_from_relative_humidity(t, prof.r.values*units.percent)
            u_p, v_p = prof.u.values * units('m/s'), prof.v.values * units('m/s')
            
            if p[0] < p[-1]: p,t,td,u_p,v_p = p[::-1],t[::-1],td[::-1],u_p[::-1],v_p[::-1]
            t[0], td[0] = s['t']*units.degC, s['td']*units.degC
            cape, cin = mpcalc.surface_based_cape_cin(p, t, td)
            
            try:
                theta = mpcalc.potential_temperature(p[:3], t[:3])
                dz = mpcalc.thickness_hydrostatic(p[:3], t[:3]) / 2.0
                N = math.sqrt(max(((G / theta[1]) * ((theta[2] - theta[0]) / dz)).magnitude, 0.0001))
            except: N = 0.01

            w_dyn = -1.0 * s['divergence'] * PBL_DEPTH.magnitude
            elev, dhx, dhy = dem.get_elevation_and_gradient(s['lat'], s['lon'])
            w_oro = (s['u']*dhx + s['v']*dhy)
            Fr = math.sqrt(s['u']**2 + s['v']**2) / (N * max(elev, 10))
            w_oro_eff = w_oro if Fr > 1.0 else w_oro * Fr
            
            parcel = KineticParcel(start_w=(w_dyn + w_oro_eff)*0.5)
            indices = {'Profile': mpcalc.parcel_profile(p, t[0], td[0]).to('degC'), 'CAPE': cape}
            path_df, drift = parcel.solve(p, t, td, u_p, v_p, indices)
            
            max_w = path_df['w'].max() if not path_df.empty else (w_dyn+w_oro_eff)*0.5

            # JSON Serileştirme Korumalı Veri Paketi
            data_to_store = {
                "Station": str(s['id']), "Timestamp": str(safe_timestamp),
                "Coordinates": {"lat": float(s['lat']), "lon": float(s['lon'])},
                "Surface": {"temp": float(s['t']), "dewpoint": float(s['td']), "wind_speed": float(s['speed']), "precip_total": float(s['precip_total'])},
                "Dynamics": {
                    "W_Dyn": float(round(w_dyn,3)), "W_Oro": float(round(w_oro_eff,3)),
                    "CAPE": float(round(cape.magnitude, 1)), "Final_Updraft": float(round(max_w,2))
                },
                "Advection": {k: float(v) for k, v in drift.items()}
            }
            
            send_to_firebase(data_to_store, safe_timestamp)
                
        except Exception as e: print(f" [!] {s['id']} hesabı atlandı: {e}")

    ds.close()
    print("=== OTONOM DÖNGÜ TAMAMLANDI ===")
