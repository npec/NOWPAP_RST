import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import numpy as np
import requests
import scipy.io as sio
from matplotlib import ticker, patches, colors
from matplotlib.path import Path as mPath
from netCDF4 import Dataset, num2date
from pyproj import Geod
from scipy import stats
from scipy.special import ndtr, ndtri
from tqdm import tqdm

SEN_DICT = {'SeaWiFS': 'S', 'MODIS-Aqua': 'A', 'MERIS': 'M', 'VIIRS-SNPP': 'V', 'YOC': 'Y', 'SGLI': 'GS'}
URL = 'https://ocean.nowpap3.go.jp/image_search/{filetype}/{subarea}/{year}/{filename}'


# ====================
# online match-up tool
# ====================
@dataclass
class ReturnValue:
    """
    Function return values
     r:  float
        correlation coefficient
     p:  float
        p-value for the test of the correlation coefficient
    """
    xp: np.array
    yp: np.array
    desc: str


def regress(x, y, scale: str = 'log-log'):
    """
    https://en.wikipedia.org/wiki/Log%E2%80%93log_plot
    :param x:
    :param y:
    :param scale:
    :return:
    """

    if scale == 'log-log':
        xi = np.log10(x)
        yi = np.log10(y)
    else:
        xi = np.asarray(x)
        yi = np.asarray(y)

    f = np.poly1d(np.polyfit(xi, yi, 1))
    m, b = f.coeffs

    if scale == 'log-log':
        xx = np.logspace(np.min(xi), np.max(xi), 100)
        yy = (xx ** m) * (10 ** b)
        bias = np.power(10, np.sum(yi - xi) / xi.size)
    else:
        xx = np.linspace(np.min(xi), np.max(xi), 100)
        yy = xx * m + b
        bias = np.sum(yi - xi) / xi.size

    r, p = stats.spearmanr(xi, yi)
    p = 0.001 if p < 0.001 else (
        0.01 if p < 0.01 else (
            0.05 if p < 0.05 else p))
    pv = f"p<{float(f'{p:.3f}')}" if p < 0.05 else f"p={float(f'{p:.3f}')}"

    if b > 0:
        s = f'+ {b:.3f}'
    else:
        s = f'{b:.3f}'

    # predict y values of origional data using the fit
    p_y = f(xi)
    # calculate the y-error (residuals)
    yp = yi - p_y
    # sum of the squares of the residuals | SSE
    sse = np.sum(np.power(yp, 2))
    # sum of the squares total | SST
    sst = np.sum(np.power(yi - yi.mean(), 2))
    # r-squared
    rsq = 1 - sse / sst

    base = r'\log_{10}'
    if scale == 'log-log':
        txt = f'${base} (y)={m:.3f} \\times {base} (x) {s}$\n' \
              f'$N={xi.size}$\n$R^{2}={rsq:.2f}$\n$r={r:.2f}$\n' \
              f'${pv}$\n$\\delta={bias:.2f}$'
        return ReturnValue(xp=xx, yp=yy, desc=txt)

    txt = f'$y={m:.3f} x {s}$\n' \
          f'$N={xi.size}$\n$R^{2}={rsq:.2f}$\n$r={r:.2f}$\n' \
          f'${pv}$\n$\\delta={bias:.2f}$'
    return ReturnValue(xp=xx, yp=yy, desc=txt)


def logticks(ax, n, axes: str = 'both'):
    # set y ticks
    subs = np.hstack((np.arange(2, 10) * 10 ** -2,
                      np.arange(2, 10) * 10 ** -1,
                      np.arange(2, 10) * 10 ** 0,
                      np.arange(2, 10) * 10 ** 1))
    major = ticker.LogLocator(base=10, numticks=5)
    minor = ticker.LogLocator(base=10, subs=subs, numticks=10)

    if axes == 'both':
        ax.xaxis.set_major_locator(major)
        ax.yaxis.set_major_locator(major)
        ax.xaxis.set_minor_locator(minor)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.yaxis.set_minor_locator(minor)
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

        ax.set_xticklabels(['' if i % n == 0 else f'{t:g}'
                            for i, t in enumerate(ax.get_xticks())])
        ax.set_yticklabels(['' if i % n == 0 else f'{t:g}'
                            for i, t in enumerate(ax.get_yticks())])

    if axes == 'x':
        ax.xaxis.set_major_locator(major)
        ax.xaxis.set_minor_locator(minor)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xticklabels(['' if i % n == 0 else f'{t:g}'
                            for i, t in enumerate(ax.get_xticks())])
    if axes == 'y':
        ax.yaxis.set_major_locator(major)
        ax.yaxis.set_minor_locator(minor)
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_yticklabels(['' if i % n == 0 else f'{t:g}'
                            for i, t in enumerate(ax.get_yticks())])
    return


def underestimated(x, y, ratio):
    idx = np.where(x > y * ratio)
    return x[idx], y[idx]


def overestimated(x, y, ratio):
    idx = np.where(y > x * ratio)
    return x[idx], y[idx]


def identity_line(ax, lines: list):
    # Add 1:1, 1:2, 1:3 or 2:1, 3:1 lines
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()

    for r in lines:
        if r == 1:
            ax.plot([x0, x1], [y0, y1], '-r', linewidth=2, label='$x = y$')
        if r == 2:
            ax.plot([x0 * r, x1], [y0, y1 / r], ':k', linewidth=2)
            ax.plot([x0, x1 / r], [y0 * r, y1], ':k', linewidth=2)
        if r == 3:
            ax.plot([x0 * r, x1], [y0, y1 / r], '-', color='#2F5597', linewidth=2)
            ax.plot([x0, x1 / r], [y0 * r, y1], '-', color='#2F5597', linewidth=2)

    ax.set_xlim([x0, x1])
    ax.set_ylim([y0, y1])
    return


def validation(x, y, plt, xlabel: str, ylabel: str, xscale: str = 'log',
               yscale: str = 'log', xlim=None, ylim=None, id_lines=None,
               s: int = 100, alpha: float = 0.4, marker: str = 'o',
               figure_name: str = None):
    if id_lines is None:
        id_lines = [1, 2, 3]
    if ylim is None:
        ylim = [0.01, 100]
    if xlim is None:
        xlim = [0.01, 100]

    fig, ax = plt.subplots(figsize=(12, 9))

    # Display scatter
    ax.scatter(
        x, y,
        c='k',
        marker=marker,
        alpha=alpha,
        s=s,
    )

    # Set axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set the scale to log
    ax.set_xscale(xscale)
    ax.set_yscale(xscale)

    # Set axis limits
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    # Identity line
    identity_line(ax=ax, lines=id_lines)

    # Configure ticks
    logticks(ax=ax, n=2)

    # overestimate
    xov, yov = overestimated(x=x.to_numpy(),
                             y=y.to_numpy(),
                             ratio=id_lines[-1])
    ax.scatter(xov, yov, s=s, c='r', marker='o', alpha=0.5)

    # underestimate
    xun, yun = underestimated(x=x.to_numpy(),
                              y=y.to_numpy(),
                              ratio=id_lines[-1])
    ax.scatter(xun, yun, s=s, c='b', marker='o', alpha=0.5)

    # Regression line
    mask = np.isnan(x.to_numpy()) | np.isnan(y.to_numpy())
    result = regress(x=x.to_numpy()[~mask].flatten(),
                     y=y.to_numpy()[~mask].flatten(),
                     scale=f'{xscale}-{yscale}')
    rl = ax.plot(result.xp, result.yp, '-g', label='linear regression', lw=6)

    plt.legend(loc='lower right')
    t = ax.text(.015, 3, result.desc)

    if figure_name:
        plt.tight_layout()
        plt.savefig(figure_name, dpi=200)
    return


def add_coastline(ax, file: Path = None, xlim=None, ylim=None):
    # we have defined a custom function that reads coastline data in util
    key = 'ncst'
    if file is None:
        print(Path('.').parent)
        file = '{}/{}'.format(Path('.').absolute().parent,
                              'sutilsx/nowpap_sea.mat')
        # print(file)
    cline = sio.loadmat(file, variable_names=[key], squeeze_me=True).get(key)

    x, y = cline[:, 0], cline[:, 1]
    ax.plot(x, y, '-k')
    if xlim is None:
        ax.set_xlim(np.nanmin(x), np.nanmax(x))
    else:
        ax.set_xlim(np.nanmin(xlim), np.nanmax(xlim))

    if ylim is None:
        ax.set_ylim(np.nanmin(y), np.nanmax(y))
    else:
        ax.set_ylim(np.nanmin(ylim), np.nanmax(ylim))
    return


def mpl_custom(mpl):
    # custom image settings
    mpl.rcParams.update({
        'font.size': 14,
        # 'axes.grid': True,
        'axes.linewidth': 2,
        'grid.linewidth': 1,

        'xtick.major.size': 8,
        'xtick.major.width': 2,

        'xtick.minor.visible': True,
        'xtick.minor.size': 4,
        'xtick.minor.width': 1,

        'ytick.major.size': 8,
        'ytick.major.width': 2,

        'ytick.minor.visible': True,
        'ytick.minor.size': 4,
        'ytick.minor.width': 1,

        'savefig.facecolor': '#F5F5F5'
    })
    return


def bilinear_interp(src_geo: np.array, interval: int):
    """
        Bilinear interpolation of SGLI geo-location corners to a spatial grid

        Parameters
        ----------
        src_geo: np.array
            either lon or lat
        interval: int
            resampling interval in pixels

        Return
        ------
        out_geo: np.array
            2-D array with dims == to geophysical variables
    """
    sds = np.concatenate((src_geo, src_geo[-1].reshape(1, -1)), axis=0)
    sds = np.concatenate((sds, sds[:, -1].reshape(-1, 1)), axis=1)

    ratio_0 = np.tile(
        np.linspace(0, (interval - 1) / interval, interval, dtype=np.float32),
        (sds.shape[0] * interval, sds.shape[1] - 1))

    ratio_1 = np.tile(
        np.linspace(0, (interval - 1) / interval, interval, dtype=np.float32).reshape(-1, 1),
        (sds.shape[0] - 1, (sds.shape[1] - 1) * interval))

    sds = np.repeat(sds, interval, axis=0)
    sds = np.repeat(sds, interval, axis=1)
    interp = (1. - ratio_0) * sds[:, :-interval] + ratio_0 * sds[:, interval:]
    return (1. - ratio_1) * interp[:-interval, :] + ratio_1 * interp[interval:, :]


def navigation_data(file: Path, key: str):
    with h5py.File(file, 'r') as hdf:
        nsl = hdf['/Image_data'].attrs['Number_of_lines'][0]
        psl = hdf['/Image_data'].attrs['Number_of_pixels'][0]
        img_size = (slice(0, nsl), slice(0, psl))

        if 'lat' in key.lower():
            # Get Latitude
            lat = get_data(h5f=hdf, key='Latitude')
            interval = hdf['/Geometry_data/Latitude'].attrs['Resampling_interval'][0]
            lat = bilinear_interp(src_geo=lat, interval=interval)[img_size]
            return lat

        # Get Longitude
        lon = get_data(h5f=hdf, key='Longitude')
        interval = hdf['/Geometry_data/Longitude'].attrs['Resampling_interval'][0]
    is_stride_180 = False
    if np.abs(np.nanmin(lon) - np.nanmax(lon)) > 180.:
        is_stride_180 = True
        lon[lon < 0] = 360. + lon[lon < 0]
    lon = bilinear_interp(src_geo=lon, interval=interval)[img_size]

    if is_stride_180:
        lon[lon > 180.] = lon[lon > 180.] - 360.
    return lon


def get_data(h5f: h5py, key: str):
    data = h5f[f'Geometry_data/{key}'][:]
    attrs = dict(h5f[f'Geometry_data/{key}'].attrs)

    if 'Error_DN' in attrs.keys():
        data[data == attrs.pop('Error_DN')[0]] = np.NaN

    if ('Minimum_valid_DN' in attrs.keys()) and \
            ('Maximum_valid_DN' in attrs.keys()):
        valid_min = attrs.pop('Minimum_valid_DN')[0]
        valid_max = attrs.pop('Maximum_valid_DN')[0]
        data[(data < valid_min) | (data > valid_max)] = np.NaN

    # Convert DN to PV
    if ('Slope' in attrs.keys()) and ('Offset' in attrs.keys()):
        data = data * attrs.pop('Slope')[0] + attrs.pop('Offset')[0]

    if ('Minimum_valid_value' in attrs.keys()) and \
            ('Maximum_valid_value' in attrs.keys()):
        valid_min = attrs.pop('Minimum_valid_value')[0]
        valid_max = attrs.pop('Maximum_valid_value')[0]
        data[(data < valid_min) | (data > valid_max)] = np.NaN

    return data


def attr_fmt(h5: h5py, address: str):
    result = {}
    for key, val in h5[address].attrs.items():
        if key in ('Dim0', 'Dim1'):
            continue
        try:
            val = val[0]
        except IndexError:
            pass

        if type(val) in (bytes, np.bytes_):
            val = val.decode()
        result.update({key: val})

    desc = result['Data_description'] \
        if 'Data_description' in result.keys() else None
    if desc and ('Remote Sensing Reflectance(Rrs)' in desc):
        result['units'] = result['Rrs_unit']
    return result


def h5_read(file: Path, key: str):
    with h5py.File(file, 'r') as h5:
        if key == 'QA_flag':
            sds = np.ma.squeeze(h5[f'Image_data/{key}'][:])
            np.ma.set_fill_value(sds, 0)
            return sds

        fill_value = np.float32(-32767)
        attrs = attr_fmt(h5=h5, address=f'Image_data/{key}')
        sdn = h5[f'Image_data/{key}'][:]

    mask = False
    if 'Error_DN' in attrs.keys():
        mask = mask | np.where(np.equal(sdn, attrs.pop('Error_DN')), True, False)
    if 'Land_DN' in attrs.keys():
        mask = mask | np.where(np.equal(sdn, attrs.pop('Land_DN')), True, False)
    if 'Cloud_error_DN' in attrs.keys():
        mask = mask | np.where(np.equal(sdn, attrs.pop('Cloud_error_DN')), True, False)
    if 'Retrieval_error_DN' in attrs.keys():
        mask = mask | np.where(np.equal(sdn, attrs.pop('Retrieval_error_DN')), True, False)
    if ('Minimum_valid_DN' in attrs.keys()) and ('Maximum_valid_DN' in attrs.keys()):
        # https://shikisai.jaxa.jp/faq/docs/GCOM-C_Products_Users_Guide_entrylevel__attach4_jp_191007.pdf#page=46
        mask = mask | np.where((sdn <= attrs.pop('Minimum_valid_DN')) |
                               (sdn >= attrs.pop('Maximum_valid_DN')), True, False)

    # Convert DN to PV
    slope, offset = 1, 0
    if 'NWLR' in key:
        if ('Rrs_slope' in attrs.keys()) and \
                ('Rrs_slope' in attrs.keys()):
            slope = attrs.pop('Rrs_slope')
            offset = attrs.pop('Rrs_offset')
    else:
        if ('Slope' in attrs.keys()) and \
                ('Offset' in attrs.keys()):
            slope = attrs.pop('Slope')
            offset = attrs.pop('Offset')

    sds = np.ma.squeeze(sdn * slope + offset)
    sds[mask] = fill_value
    sds = np.ma.masked_where(mask, sds).astype(np.float32)
    np.ma.set_fill_value(sds, fill_value)
    return sds


# ====================
# Time-series analysis
# ====================

@dataclass
class LSF:
    """
    Class for keeping track of the return values
     m:  float
        slope
     b:  float
        y-intercept
     p:  float
        p-value for the test of the slope
     t_stat: float
        t-statist from observations
     t_crit: float
        t-critical from the theoretical expectation
    """
    m: float
    b: float
    p: float
    t_stat: float
    t_crit: float


@dataclass
class MKT:
    """
    Class for keeping track of the return values
     m:  float
        slope
     b:  float
        y-intercept
     p:  float
        p-value for the test of the slope
     z_score: float
        z-statist from observations
     z_crit: float
        z-critical from the theoretical expectation
    """
    m: float
    b: float
    p: float
    z_score: float
    z_crit: float


def add2map(file: Path, ax, point: dict = None, region: dict = None):
    lat = nc_reader(file=file, var='lat')
    lon = nc_reader(file=file, var='lon')
    add_coastline(
        ax=ax
        , xlim=[np.min(lon), np.max(lon)]
        , ylim=[np.min(lat), np.max(lat)])

    if point is not None:
        x, y = point['lon'][0], point['lat'][0]
        ax.scatter(x, y, s=200, c='orange', edgecolors='k')

    if region is not None:
        path = mpl_path(bbox=region)
        patch = patches.PathPatch(path, facecolor='orange', lw=2)
        ax.add_patch(patch)
    return


def get_nc_keys(file: Path):
    exclude = 'lon', 'lat', 'crs', 'time'
    with Dataset(file, 'r') as nc:
        keys = [key for key in nc.variables.keys()
                if (key not in exclude)]
    return keys


def mpl_path(bbox: dict):
    x0, x1 = bbox['lon']
    y0, y1 = bbox['lat']

    # path vertex coordinates
    vertices = [
        (x0, y0),  # left, bottom
        (x0, y1),  # left, top
        (x1, y1),  # right, top
        (x1, y0),  # right, bottom
        (x0, y0),  # ignored
    ]

    codes = [
        mPath.MOVETO,
        mPath.LINETO,
        mPath.LINETO,
        mPath.LINETO,
        mPath.CLOSEPOLY,
    ]

    return mPath(vertices, codes)


def area_mask(bbox: dict, lon, lat):
    path = mpl_path(bbox=bbox)

    # create a mesh grid for the whole image
    if len(lon.shape) == 1:
        x, y = np.meshgrid(lon, lat)
    else:
        x, y = lon, lat
    # mesh grid to a list of points
    points = np.vstack((x.ravel(), y.ravel())).T

    # select points included in the path
    mask = path.contains_points(points)
    return np.array(mask).reshape(x.shape)


def harv_dist(px: float, py: float, lon, lat, datum: str = 'WGS84'):
    """
    Distance from a point x, y in geographical space
    """
    p = np.pi / 180.
    g = Geod(ellps=datum)

    a = (g.a ** 2 * np.cos(py * p)) ** 2
    b = (g.b ** 2 * np.sin(py * p)) ** 2

    c = (a * np.cos(py * p)) ** 2
    d = (b * np.cos(py * p)) ** 2

    r = np.sqrt((a + b) / (c + d))

    if len(lon.shape) == 1:
        x, y = np.meshgrid(lon, lat)
    else:
        x, y = lon, lat

    px = np.ones(x.shape) * px
    py = np.ones(x.shape) * py
    e = 0.5 - np.cos((y - py) * p) / 2 + np.cos(
        y * p) * np.cos(py * p) * (
                1 - np.cos((px - x) * p)) / 2
    # 2*R*asin..
    return 2 * r * np.arcsin(e ** .5)


def fmt_sds(sds, mask):
    result, true = [], np.bool_(True)
    for v, m in zip(sds, mask):
        if m is true:
            result.append('-999')
            continue
        result.append(f'{v:.8f}')
    return result


def pyextract(bbox: dict, file_list: list, filename: Path, window: int = None):
    lat = nc_reader(file_list[0], 'lat')
    lon = nc_reader(file_list[0], 'lon')

    def extract():
        masked = sds[mask]
        valid = np.ma.compressed(masked)
        valid_px = valid.size
        total_px = masked.size
        invalid_px = total_px - valid_px

        if np.all(masked.mask):
            dset = [f.name, tcs, tce, total_px, valid_px, invalid_px, fill_value,
                    fill_value, fill_value, fill_value, fill_value, fill_value]
            if len(bbox['lon']) == 2:
                dset = dset[:-1]
            fmt = ['s', 's', 's', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g']
        else:
            sds_mean = 10 ** np.log10(valid).mean()
            sds_std = 10 ** np.log10(valid).std()
            sds_max = valid.max()
            sds_min = valid.min()
            sds_med = np.median(valid)
            dset = [f.name, tcs, tce, total_px, valid_px, invalid_px,
                    sds_min, sds_max, sds_mean, sds_med, sds_std]
            fmt = ['s', 's', 's', 'g', 'g', 'g', '.6f', '.6f', '.6f', '.6f', '.6f', '.6f']
            if len(bbox['lon']) == 1:
                pxv = sds[px_value][0] if sds[px_value].mask[0] is np.bool_(False) else fill_value
                dset.append(pxv)
                if pxv == -999:
                    fmt = ['s', 's', 's', 'g', 'g', 'g', '.6f', '.6f', '.6f', '.6f', '.6f', 'g']

        line = ','.join([f'{val:{sf}}' for val, sf in zip(dset, fmt)])
        txt.writelines(f'{line}\n')

    with open(filename, 'w') as txt:
        if len(bbox['lon']) == 1:
            px, py = bbox['lon'][0], bbox['lat'][0]

            txt.writelines('filename,time_start,time_end,pixel_count,valid,invalid,'
                           'min,max,mean,median,std,pixel_value\n')

            dist = harv_dist(px=px, py=py, lon=lon, lat=lat)
            px_value = np.where(dist == dist.min())
            mask = np.zeros_like(dist)
            mask[px_value] = 1

            if window:
                if window % 2 == 0:
                    raise Exception('Window must be an odd number!!')
                kernel = np.ones((window, window), np.uint8)
                mask = np.bool_(cv2.dilate(mask, kernel, iterations=1))
            else:
                mask = np.bool_(mask)

        elif len(bbox['lon']) == 2:
            mask = area_mask(bbox=bbox, lon=lon, lat=lat)
            txt.writelines('filename,time_start,time_end,pixel_count,valid,invalid,'
                           'min,max,mean,median,std\n')

        else:
            raise Exception('Unexpected number of values in the BBOX')

        fill_value = -999
        if len(file_list) > 1:
            for i, f in enumerate(file_list):
                sds = nc_reader(file=f, var='chlor_a')
                tcs = nc_attribute(file=f, name='time_coverage_start')
                tce = nc_attribute(file=f, name='time_coverage_end')

                extract()

        if len(file_list) == 1:
            f = file_list[0]
            for i, key in enumerate(get_nc_keys(file=f)):
                sds = nc_reader(file=f, var=key)
                try:
                    tcs = nc_attribute(file=f, name='time_coverage_start')
                    tce = nc_attribute(file=f, name='time_coverage_end')
                except AttributeError:
                    try:
                        sy = nc_attribute(file=f, name='Start Year', location=key)
                        sd = nc_attribute(file=f, name='Start Day', location=key)
                        tcs = datetime.strptime(f'{sy}{sd:03}', '%Y%j').strftime('%FT%H:%M:%SZ')

                        ey = nc_attribute(file=f, name='End Year', location=key)
                        ed = nc_attribute(file=f, name='End Day', location=key)
                        tce = datetime.strptime(f'{ey}{ed:03}', '%Y%j').strftime('%FT%H:%M:%SZ')

                    except AttributeError:
                        tcs = tce = '-999'
                extract()
    return


def preallocate(file: Path, varname: str, t: int):
    with Dataset(file, 'r') as nc:
        shape = np.ma.squeeze(nc[varname][:]).shape
        dtype = nc[varname][:].dtype
    shape = (t,) + shape
    return np.ma.empty(shape=shape, dtype=dtype)


def get_min_max(files: list, varname: str = 'chlor_a', case: str = 'max'):
    sds = preallocate(file=files[0], t=len(files), varname=varname)
    for j, f in enumerate(files):
        sds[j, :, :] = nc_reader(file=f, var=varname)
    fill_value = nc_reader(file=files[0], var=varname).fill_value

    input_dates = []
    append = input_dates.append

    for j, f in enumerate(files):
        append(nc_reader(file=f, var='time')[0])

    mask = sds.mean(axis=0).mask
    # Min data
    if case == 'min':
        data = np.ma.amin(sds, axis=0)
        data = np.ma.masked_where(mask, data)
        np.ma.set_fill_value(data, fill_value=fill_value)

        idx = np.ma.argmin(sds, axis=0)
        doy = np.array([d.timetuple().tm_yday for d in input_dates])[idx]
        doy = np.ma.masked_where(idx == 0, doy)
        return data, doy

    # Max data
    mask = sds.mean(axis=0).mask
    data = np.ma.amax(sds, axis=0)
    data = np.ma.masked_where(mask, data)
    np.ma.set_fill_value(data, fill_value=fill_value)

    idx = np.ma.argmax(sds, axis=0)
    doy = np.array([d.timetuple().tm_yday for d in input_dates])[idx]
    doy = np.ma.masked_where(idx == 0, doy)

    return data, doy


def nc_attribute(file: Path, name: str, location: str = '/'):
    with Dataset(file, 'r') as nc:
        if location == '/':
            return nc.getncattr(name)
        value = nc[location].getncattr(name)
    return value


def nc_reader(file: Path, var: str):
    with Dataset(file, 'r') as nc:
        if var == 'time':
            sds = num2date(nc[var][:], units=nc[var].units)
        else:
            sds = np.ma.squeeze(nc[var][:])
    return sds


def lsq_fity(x, y, alpha: float = 0.05):
    """
     Calculate a "MODEL-1" least squares fit by:  Edward T Peltzer, MBARI

     The line is fit by MINIMIZING the residuals in Y only.

     The equation of the line is:     Y = my * X + by.

     Equations are from Bevington & Robinson (1992)
       Data Reduction and Error Analysis for the Physical Sciences, 2nd Ed."
       pp: 104, 108-109, 199.

     Data are input and output as follows:
         m, b, r, sm, sb = lsq_fity(x, y)

    Parameters
    ----------
    x: np.array
        input x vector
    y: np.array
        input y vector
    alpha: float

    Returns
    -------
    ReturnValue:
        return values in dataclass
    """

    x = np.asarray(x)
    y = np.asarray(y)
    # Determine the size of the vector
    n = len(x)

    # Calculate the sums
    sx = x.sum()
    sy = y.sum()
    sx2 = np.sum(x ** 2)
    sxy = np.sum(x * y)

    # Calculate re-used expressions
    num = n * sxy - sx * sy
    den = n * sx2 - sx ** 2

    # Calculate my, by, ry, s2, smy and sby
    my = num / den
    by = (sx2 * sy - sx * sxy) / den

    df = n - 2
    diff = y - by - my * x

    s2 = np.sum(diff * diff) / (n - 2)
    smy = np.sqrt(n * s2 / den)

    t_stat = my / smy
    t_crit = stats.t.ppf(1 - alpha / 2, df)  # equivalent to Excel TINV(0.05, DF)
    py = stats.t.sf(t_stat, df=df) * 2  # two-sided p-value = Prob(abs(t)>tt)

    return LSF(my, by, py, t_stat, t_crit)


def mktest(x, y, eps=1e-6, alpha: float = 0.1):
    """
        Runs the Mann-Kendall test for trend in time series data.
        https://up-rs-esp.github.io/mkt/

        Parameters
        ----------
        x : 1D numpy.ndarray
            array of the time points of measurements
        y : 1D numpy.ndarray
            array containing the measurements corresponding to entries of 't'
        eps : scalar, float, greater than zero
            least count error of measurements which help determine ties in the data
        alpha:

        Returns
        -------
        m : scalar, float
            slope of the linear fit to the data
        c : scalar, float
            intercept of the linear fit to the data
        p : scalar, float, greater than zero
            p-value of the obtained Z-score statistic for the Mann-Kendall test

        """

    x = np.asarray(x)
    y = np.asarray(y)

    # estimate sign of all possible (n(n-1)) / 2 differences
    n = len(x)
    zscore = None
    sgn = np.zeros((n, n), dtype=np.int32)
    for i in range(n):
        tmp = y - y[i]
        tmp[np.where(np.fabs(tmp) <= eps)] = 0
        sgn[i] = np.sign(tmp)

    # estimate mean of the sign of all possible differences
    # s = sgn[np.triu_indices(n, k=1)].sum()
    s = 0
    for j in range(n):
        tmp = y[(j + 1):] - y[j]
        tmp[np.where(np.fabs(tmp) <= eps)] = 0
        s += np.sign(tmp).sum()

    # estimate variance of the sign of all possible differences
    # 1. Determine no. of tie groups 'p' and no. of ties in each group 'q'
    np.fill_diagonal(sgn, eps * 1E6)
    i, j = np.where(sgn == 0.)
    ties = np.unique(y[i])
    p = len(ties)
    q = np.zeros(len(ties), dtype=np.int32)
    for k in range(p):
        idx = np.where(np.fabs(y - ties[k]) < eps)[0]
        q[k] = len(idx)

    # 2. Determine the two terms in the variance calculation
    term1 = n * (n - 1) * (2 * n + 5)
    term2 = (q * (q - 1) * (2 * q + 5)).sum()
    # 3. estimate variance
    var_s = float(term1 - term2) / 18.

    # Compute the Z-score and the p-value for the obtained Z-score
    if s > eps:
        zscore = (s - 1) / np.sqrt(var_s)
        p = 0.5 * (1. - ndtr(zscore))
    elif np.fabs(s) <= eps:
        zscore, p = 0., 0.5
    elif s < -eps:
        zscore = (s + 1) / np.sqrt(var_s)
        p = 0.5 * (ndtr(zscore))

    # compute test based on given 'alpha' and alternative hypothesis
    z_crit = ndtri(1. - alpha / 2.)

    # estimate the slope and intercept of the line
    m = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
    c = np.mean(y) - m * np.mean(x)

    return MKT(m, c, p, zscore, z_crit)


def sen_slope(x, y):
    """
     Calculate a "MODEL-1" least squares fit by:  Edward T Peltzer, MBARI

     The line is fit by MINIMIZING the residuals in Y only.

     The equation of the line is:     Y = my * X + by.

     Equations are from Bevington & Robinson (1992)
       Data Reduction and Error Analysis for the Physical Sciences, 2nd Ed."
       pp: 104, 108-109, 199.

     Data are input and output as follows:
         m, b, r, sm, sb = lsq_fity(x, y)

    Parameters
    ----------
    x: np.array
        input x vector
    y: np.array
        input y vector

    Returns
    -------
    ReturnValue:
        return values in dataclass
    """
    x = np.asarray(x)
    y = np.asarray(y)

    n = x.size
    ns = int(n * (n - 1) / 2)
    slope = np.empty((ns,), dtype=np.float32)

    start, end = 0, 0
    for i in range(n):
        tmp = (y[(i + 1):] - y[i]) / (x[(i + 1):] - x[i])
        end += tmp.size
        slope[start:end] = tmp
        start = end
    return np.median(slope)


# Day month fetching file generator
def dmo_filegen(sensor: str, year: int, start_month: int, end_month: int, filetype: tuple,
                composite_period: str, start_day: int, end_day: int, variable: str, subarea: str):
    init = SEN_DICT[sensor]
    # Define the netCDF (PNG) file name
    end_mo = end_month + 1 if end_month < 13 else end_month
    for month in range(start_month, end_mo):
        if composite_period == 'day':
            end_dy = datetime(year, month + 1, 1) if month < 12 else datetime(year + 1, 1, 1)
            end_dy = min(datetime.fromordinal(end_dy.toordinal() - 1).day, end_day)

            for day in range(start_day, end_dy + 1):
                files = [f'{init}{year}{month:02}{day:02}_{variable}_{subarea}_{composite_period}.{ext}'
                         for ext in filetype]

                yield from [URL.format(filetype='netcdf', subarea=subarea, year=year, filename=f)
                            if f.endswith('.nc') else
                            URL.format(filetype='images', subarea=subarea, year=year, filename=f)
                            for f in files]

        if composite_period == 'month':
            files = [f'{init}{year}{month:02}_{variable}_{subarea}_{composite_period}.{ext}'
                     for ext in filetype]

            yield from [URL.format(filetype='netcdf', subarea=subarea, year=year, filename=f)
                        if f.endswith('.nc') else
                        URL.format(filetype='images', subarea=subarea, year=year, filename=f)
                        for f in files]


# Function to download the data
def get_file(query_url: str, opath: Path, bar: tqdm):
    # path = output_dir.format(sen=sen)
    # path = output_dir
    basename = os.path.basename(query_url)

    # if not os.path.isdir(path):
    #     os.makedirs(path)

    savefile = opath.joinpath(basename)
    size = 0
    if savefile.is_file():
        size = savefile.stat()

    time.sleep(0.1)
    with requests.get(query_url) as r:
        if r.status_code != 200:
            bar.set_description(f'FileNotFound: {basename}')
            bar.update()
            return
        total = int(r.headers.get('content-length'))
        if total == size:
            bar.set_description(f'FileExists, Skip: {basename}')
            bar.update()
            return

            # print('File: {} '.format(savefile), end='')
        bar.set_description(f'Downloading: {basename}')
        bar.update()
        with open(savefile, "wb") as handle:
            for chunk in r.iter_content(chunk_size=max(int(total / 1000), 1024 * 1024)):
                # download progress check tqdm
                if chunk:
                    handle.write(chunk)
    return


def download(variable: str, subarea: str, sensor: str, start_year: int,
             end_year: int, composite_period: str, start_month: int, file_type: tuple,
             end_month: int, start_day: int, end_day: int, output_dir: Path):
    if composite_period == 'year':
        total = len(range(start_year, end_year + 1))
    else:
        if composite_period == 'month':
            total = len([datetime(y, m, 1)
                         for y in range(start_year, end_year + 1)
                         for m in range(1, 13)
                         if (y <= end_year) and (m <= end_month)])
        else:
            start = datetime(start_year, start_month, start_day).toordinal()
            try:
                end = datetime(end_year, end_month, end_day).toordinal() + 1
            except ValueError:
                end = datetime(end_year + 1, 1, 1).toordinal() \
                    if (end_month == 12) else datetime(end_year, end_month + 1, 1).toordinal()
            total = len(range(start, end))

    with tqdm(total=total) as bar:
        for year in range(start_year, end_year + 1):
            if composite_period in ('day', 'month'):
                for query in dmo_filegen(
                        sensor=sensor
                        , year=year
                        , filetype=file_type
                        , composite_period=composite_period
                        , variable=variable
                        , start_month=start_month
                        , end_month=end_month
                        , start_day=start_day
                        , end_day=end_day
                        , subarea=subarea):
                    # --------------------------------------------------
                    get_file(query_url=query, opath=output_dir, bar=bar)
                    # --------------------------------------------------

            if composite_period == 'year':
                init = SEN_DICT[sensor]
                for ext in file_type:
                    filetype = 'netcdf' if 'nc' in ext else 'images'
                    # -----------------------------------------------------------------
                    file = f'{init}{year}_{variable}_{subarea}_{composite_period}.{ext}'
                    query = URL.format(filetype=filetype,
                                       subarea=subarea, year=year, filename=file)
                    # -----------------------------------------------------------
                    get_file(query_url=query, opath=output_dir, bar=bar)
                    # --------------------------------------------------
    return


def nc_write(file: Path, data, varname: str, lon, lat, count=None):
    """Caller method for writing the netcdf file"""
    basename = file.name
    start = time.perf_counter()

    with Dataset(file, 'w') as trg:
        # Global attributes
        trg.setncatts({'product_name': basename,
                       'Creator': 'NOWPAP-CEARAC (NOWPAP Training)',
                       'date_created': "{}".format(time.ctime())})

        # Create the dimensions of the file
        trg.createDimension('lat', lat.size)
        nc_dim = trg.createVariable('lat', 'float32', ('lat',))
        nc_dim.setncatts({'standard_name': 'latitude',
                          'long_name': 'latitude',
                          'units': 'degrees_north',
                          'axis': 'Y',
                          'valid_min': lat.min(),
                          'valid_max': lat.max()})
        nc_dim[:] = lat

        trg.createDimension('lon', lon.size)
        nc_dim = trg.createVariable('lon', 'float32', ('lon',))
        nc_dim.setncatts({'standard_name': 'longitude',
                          'long_name': 'longitude',
                          'units': 'degrees_east',
                          'axis': 'Y',
                          'valid_min': lon.min(),
                          'valid_max': lon.max()})
        nc_dim[:] = lon

        comp = trg.createVariable(varname, 'float32',
                                  (u'lat', u'lon'),
                                  zlib=True, complevel=6)
        comp.setncatts({
            'long_name': 'concentration_of_phytoplankton_green_pigment_in_surface_water'
            , 'standard_name': 'Chlorophyll-a concentration'
            , 'units': 'mg/m^3'
            , '_FillValue': data.fill_value.astype(np.float32)
            , 'valid_min': data.min().astype(np.float32)
            , 'valid_max': data.max().astype(np.float32)
        })
        comp[:] = data

        if count is not None:
            """Creates the valid pixel count for the composite data"""
            #nc_dim = trg.createVariable('count', 'int16',
            nc_dim = trg.createVariable('valid_pixel_count', 'int16',
                                        (u'lat', u'lon'),
                                        zlib=True, complevel=6)
            nc_dim.setncatts({
                'standard_name': 'count of valid pixels'
                , 'long_name': 'number of valid data in each pixel for the composite period'
                , 'units': 'count'
                , '_FillValue': np.int16(count.fill_value)
                , 'valid_min': count.min().astype('int16')
                , 'valid_max': count.max().astype('int16')})
            nc_dim[:] = count

    elapsed = time.perf_counter() - start
    hour = int(elapsed // 3600)
    mnt = int(elapsed % 3600 // 60)
    sec = elapsed % 3600 % 60

    f = basename.strip('.nc')
    print(f'NCWRITE: {f} | Elapsed: {hour:2} hours {mnt:2} minutes {sec:5.3f} seconds')
    return


def get_cmap():
    rr = [0, 191, 190]
    gg = [24, 171, 0]
    bb = [190, 4, 24]

    rr, gg, bb = (np.array(rr) / 255., np.array(gg) / 255., np.array(bb) / 255.)

    colours = list()
    for r, g, b in zip(rr, gg, bb):
        colours.append([r, g, b])

    n = np.array(colours).shape[0]
    # percent level of each colour in the colour map
    levels = (np.array(range(n)) / float(n - 1)).tolist()  # type: list

    stop = 0
    r, g, b = [], [], []
    while stop < n:
        r.append(tuple([levels[stop], colours[stop][0], colours[stop][0]]))
        g.append(tuple([levels[stop], colours[stop][1], colours[stop][1]]))
        b.append(tuple([levels[stop], colours[stop][2], colours[stop][2]]))
        stop += 1
    colour_map = {
        'red': tuple(r),
        'green': tuple(g),
        'blue': tuple(b)
    }
    return colors.LinearSegmentedColormap('slope', colour_map)