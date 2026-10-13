"""AM regional NDVI (GIMMS NDVI3g, 1982-2022): monthly mean over the AM bounding rectangle, STL residuals,
and rolling 120-month AR1 and DFA used for Figs. 4 and 5."""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import netCDF4 as nc
import geopandas as gpd
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from statsmodels.tsa.seasonal import STL
import fathon
from fathon import fathonUtils as fu
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
SOURCE = Path(r'E:\research_data\global_ndvi')
BOUNDARY = Path(r'D:\GEE学习\陆气耦合计算\亚马逊shp\Amazon_test.shp')


def main():
    files = sorted(SOURCE.glob('*.nc4'))
    geo = gpd.read_file(BOUNDARY).to_crs(4326)
    geo.to_file(OUT / 'am_boundary.geojson', driver='GeoJSON')
    west, south, east, north = geo.total_bounds
    with nc.Dataset(files[0]) as ds:
        lat, lon = ds['lat'][:], ds['lon'][:]
    iy = np.flatnonzero((lat >= south) & (lat <= north))
    ix = np.flatnonzero((lon >= west) & (lon <= east))
    ys, xs = slice(iy[0], iy[-1] + 1), slice(ix[0], ix[-1] + 1)
    lat, lon = lat[ys], lon[xs]
    transform = from_origin(lon[0] - 1 / 24, lat[0] + 1 / 24, 1 / 12, 1 / 12)
    mask = np.ones((len(lat), len(lon)), dtype=bool)  # No polygon or vegetation mask
    clip = OUT / 'am_ndvi_halfmonthly_1982_2022.nc'
    monthly = []
    dates = []
    rows = []
    cursor = 0
    with nc.Dataset(clip, 'w') as dst:
        dst.createDimension('time', None)
        dst.createDimension('lat', len(lat))
        dst.createDimension('lon', len(lon))
        for name, vals, units in [('lat', lat, 'degrees_north'), ('lon', lon, 'degrees_east')]:
            v = dst.createVariable(name, 'f8', (name,))
            v[:] = vals
            v.units = units
        tv = dst.createVariable('time', 'f8', ('time',))
        tv.units = 'days since 1982-01-01'
        tv.calendar = 'standard'
        nv = dst.createVariable(
            'ndvi', 'i2', ('time', 'lat', 'lon'), zlib=True, complevel=4, fill_value=-32768
        )
        nv.units = '1'
        nv.scale_factor = 0.0001
        nv.set_auto_maskandscale(False)
        qv = dst.createVariable(
            'percentile_with_flag',
            'i2',
            ('time', 'lat', 'lon'),
            zlib=True,
            complevel=4,
            fill_value=-32768,
        )
        qv.description = (
            'Original percentile: [0,1000]+2000*flag; 0 observed, 1 spline, 2 seasonal, 3 missing'
        )
        mv = dst.createVariable('am_mask', 'u1', ('lat', 'lon'))
        mv[:] = mask
        dst.boundary = str(BOUNDARY)
        dst.mask_rule = 'all pixel centers inside original AM bounding rectangle'
        dst.source = str(SOURCE)
        for fi, p in enumerate(files):
            with nc.Dataset(p) as ds:
                ds.set_auto_maskandscale(False)
                assert np.allclose(ds['lat'][ys], lat) and np.allclose(ds['lon'][xs], lon)
                raw = ds['ndvi'][:, ys, xs].copy()
                qa = ds['percentile'][:, ys, xs].copy()
                times = nc.num2date(ds['time'][:], ds['time'].units, calendar=ds['time'].calendar)
                timestamps = pd.DatetimeIndex([pd.Timestamp(t.year, t.month, t.day) for t in times])
                invalid = (
                    (raw == -32768)
                    | (raw == -5000)
                    | (raw < -3000)
                    | (raw > 10000)
                    | (~mask[None, :, :])
                )
                raw[invalid] = -32768
                qa[:, ~mask] = -32768
                n = len(times)
                nv[cursor : cursor + n] = raw
                qv[cursor : cursor + n] = qa
                tv[cursor : cursor + n] = nc.date2num(times, tv.units)
                cursor += n
                val = raw.astype('f4') / 10000
                val[invalid] = np.nan
                for month in sorted(set(timestamps.to_period('M'))):
                    take = timestamps.to_period('M') == month
                    assert take.sum() == 2
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        composite = np.nanmedian(val[take], axis=0)
                    monthly.append(composite)
                    dates.append(month.to_timestamp())
                    valid = np.isfinite(composite)
                    rows.append(
                        {
                            'date': str(month),
                            'ndvi_mean': float(np.nanmean(composite, dtype='f8')),
                            'valid_pixels': int(valid.sum()),
                            'rectangle_pixels': int(mask.sum()),
                            'coverage': float(valid.sum() / mask.sum()),
                            'flag0_fraction': float(
                                ((qa[take] >= 0) & (qa[take] <= 1000) & ~invalid[take]).sum()
                                / max((~invalid[take]).sum(), 1)
                            ),
                        }
                    )
            print(f'Clipped {fi+1}/{len(files)}: {p.name}', flush=True)
    dates = pd.DatetimeIndex(dates)
    assert dates.equals(pd.date_range('1982-01-01', '2022-12-01', freq='MS'))
    cube = np.stack(monthly)
    with nc.Dataset(OUT / 'am_ndvi_monthly_1982_2022.nc', 'w') as dst:
        for name, size in [('time', len(dates)), ('lat', len(lat)), ('lon', len(lon))]:
            dst.createDimension(name, size)
        dst.createVariable('lat', 'f8', ('lat',))[:] = lat
        dst['lat'].units = 'degrees_north'
        dst.createVariable('lon', 'f8', ('lon',))[:] = lon
        dst['lon'].units = 'degrees_east'
        tv = dst.createVariable('time', 'f8', ('time',))
        tv.units = 'days since 1982-01-01'
        tv[:] = nc.date2num(dates.to_pydatetime(), tv.units)
        v = dst.createVariable(
            'ndvi', 'f4', ('time', 'lat', 'lon'), zlib=True, complevel=4, fill_value=-9999
        )
        v[:] = np.where(np.isfinite(cube), cube, -9999)
        v.units = '1'
        dst.description = 'Calendar-month pixelwise median of two half-month composites; AM bounding rectangle, no polygon mask; no extra vegetation/QA filter'
    regional = pd.DataFrame(rows)
    x = regional.ndvi_mean.to_numpy()
    assert np.isfinite(x).all()
    stl = STL(x, period=12, robust=False).fit()
    r = stl.resid
    regional['stl_trend'] = stl.trend
    regional['stl_seasonal'] = stl.seasonal
    regional['stl_residual'] = r
    regional.to_csv(OUT / 'am_regional_monthly.csv', index=False)
    ews = []
    for i in range(len(r) - 120 + 1):
        w = r[i : i + 120]
        a, b = w[:-1], w[1:]
        ar = float(np.corrcoef(a, b)[0, 1])
        # DFA order 3 on the residual window passed directly (no cumulative integration), scales 10..120.
        d = fathon.DFA(w)
        d.computeFlucVec(fu.linRangeByStep(10, 120), revSeg=True, polOrd=3)
        ews.append(
            {
                'window_start': dates[i].strftime('%Y-%m'),
                'window_end': dates[i + 119].strftime('%Y-%m'),
                'ar1': ar,
                'dfa': float(d.fitFlucVec()[0]),
            }
        )
    ews = pd.DataFrame(ews)
    ews.to_csv(OUT / 'am_ar1_dfa_curves.csv', index=False)
    plt.rcParams.update(
        {
            'font.family': 'DejaVu Sans',
            'font.size': 10,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'svg.fonttype': 'none',
        }
    )
    fig, axs = plt.subplots(4, 1, figsize=(11, 10), sharex=True, layout='constrained')
    axs[0].plot(dates, x, color='#32704e', lw=0.9, label='Monthly regional NDVI')
    axs[0].plot(dates, stl.trend, color='#111111', lw=1.2, label='STL trend')
    axs[0].set_ylabel('NDVI')
    axs[0].legend(loc='lower left', ncol=2, frameon=False)
    axs[1].plot(dates, r, color='#666666', lw=0.8)
    axs[1].set_ylabel('STL residual')
    t = pd.to_datetime(ews.window_end)
    axs[2].plot(t, ews.ar1, color='#2367a0', label='Pearson lag-1')
    axs[2].set_ylabel('AR1')
    axs[2].legend(frameon=False)
    axs[3].plot(t, ews.dfa, color='#b54b38', label='DFA, order 3')
    axs[3].set_ylabel('DFA exponent')
    axs[3].legend(frameon=False)
    for ax in axs:
        ax.grid(alpha=0.2)
        ax.axvline(pd.Timestamp('2015-01-01'), color='#888888', lw=0.8, ls=':')
    axs[3].set_xlabel('Year (EWS dated at the final month of each 120-month window)')
    fig.suptitle(
        'AM bounding rectangle, no polygon mask | GIMMS NDVI3g | 1982–2022\nMonthly pixel median → regional mean → STL → rolling AR1 / DFA',
        fontsize=13,
    )
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(OUT / f'am_ndvi_ar1_dfa.{ext}', dpi=220)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 5), layout='constrained')
    im = ax.imshow(
        np.nanmean(cube, axis=0),
        extent=[lon[0] - 1 / 24, lon[-1] + 1 / 24, lat[-1] - 1 / 24, lat[0] + 1 / 24],
        vmin=0,
        vmax=1,
        cmap='YlGn',
    )
    geo.boundary.plot(ax=ax, color='black', linewidth=0.6)
    fig.colorbar(im, ax=ax, label='Mean NDVI, 1982–2022')
    ax.set(
        xlabel='Longitude',
        ylabel='Latitude',
        title='AM bounding rectangle (polygon outline for reference only)',
    )
    fig.savefig(OUT / 'am_clip_map.png', dpi=180)
    plt.close(fig)
    print('Done:', len(x), 'months,', len(ews), 'EWS windows')


if __name__ == '__main__':
    main()
