"""Prepare an hourly Jena dataset without inventing observations or crossing gaps."""
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
URL = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip'
INPUT_HOURS = HORIZON = 168


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def valid_starts(values, start, end, width=336):
    """Every input AND target timestamp must belong to the same split and be observed."""
    bad = (~np.isfinite(values).all(axis=1)).astype(np.int64)
    cumulative = np.r_[0, bad.cumsum()]
    candidates = np.arange(start, max(start, end - width + 1), dtype=np.int64)
    return candidates[(cumulative[candidates + width] - cumulative[candidates]) == 0]


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    archive = DATA / 'jena_climate_2009_2016.csv.zip'
    csv = DATA / 'jena_climate_2009_2016.csv'
    if not archive.exists():
        response = requests.get(URL, timeout=120)
        response.raise_for_status()
        tmp = archive.with_suffix('.part')
        tmp.write_bytes(response.content)
        tmp.replace(archive)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None, 'ZIP CRC failure'
        content = z.read(csv.name)
        if not csv.exists():
            csv.write_bytes(content)
        assert csv.read_bytes() == content, 'Local CSV differs from archive'
    source_path = DATA / 'source.json'
    source = json.loads(source_path.read_text(encoding='utf-8')) if source_path.exists() else {}
    source.update(url=URL, source_page='https://www.tensorflow.org/tutorials/structured_data/time_series',
                  original_provider='Max Planck Institute for Biogeochemistry, Jena',
                  original_provider_url='https://weather.bgc-jena.mpg.de/weather_data.html',
                  archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                  csv_sha256=hashlib.sha256(content).hexdigest(), archive_bytes=archive.stat().st_size)
    save_json(source_path, source)
    raw = pd.read_csv(csv)
    times = pd.to_datetime(raw['Date Time'], format='%d.%m.%Y %H:%M:%S')
    duplicate_count = int(times.duplicated().sum())
    # Only collapse byte-equivalent parsed records; conflicting duplicates require investigation.
    unique = raw.drop_duplicates()
    assert not unique['Date Time'].duplicated().any(), 'Conflicting duplicate timestamps'
    unique = unique.copy()
    unique.index = pd.to_datetime(unique.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')
    unique = unique.sort_index()
    hourly = unique.loc[(unique.index.minute == 0) & (unique.index.second == 0)].copy()
    grid = pd.date_range(hourly.index.min(), hourly.index.max(), freq='h')
    hourly = hourly.reindex(grid)
    missing = hourly['T (degC)'].isna()
    missing_table = pd.DataFrame({'timestamp': grid[missing].astype(str)})
    missing_table.to_csv(DATA / 'missing_hours.csv', index=False)
    sentinels = {}
    for col in ['wv (m/s)', 'max. wv (m/s)']:
        sentinels[col] = int((hourly[col] == -9999).sum())
        hourly.loc[hourly[col] == -9999, col] = 0.0
    assert not (hourly == -9999).any().any(), 'Unhandled sentinel'
    direction = np.deg2rad(hourly.pop('wd (deg)'))
    for name, prefix in [('wv (m/s)', ''), ('max. wv (m/s)', 'max ' )]:
        speed = hourly.pop(name)
        hourly[prefix + 'Wx'] = speed * np.cos(direction)
        hourly[prefix + 'Wy'] = speed * np.sin(direction)
    # Treat source timestamps as a fixed civil-time axis. Do not infer/localize DST.
    seconds = (grid - pd.Timestamp('1970-01-01')).total_seconds().to_numpy()
    for name, period in [('Day', 86400), ('Year', 365.2425 * 86400)]:
        hourly[name + ' sin'] = np.sin(2 * np.pi * seconds / period)
        hourly[name + ' cos'] = np.cos(2 * np.pi * seconds / period)
    n = len(hourly)
    boundaries = {'train': (0, int(n * .7)), 'val': (int(n * .7), int(n * .9)), 'test': (int(n * .9), n)}
    # All statistics come ONLY from observed training rows, before window creation.
    train = hourly.iloc[:boundaries['train'][1]].dropna()
    mean, std = train.mean(), train.std(ddof=0)
    assert (std > 0).all()
    values = ((hourly - mean) / std).to_numpy(dtype=np.float32)
    arrays = {'values': values, 'time_seconds': seconds.astype(np.int64)}
    split_info = {}
    for name, (lo, hi) in boundaries.items():
        starts = valid_starts(values, lo, hi)
        arrays[name + '_starts'] = starts
        assert len(starts) > 0
        # Independent checks against real timestamps and missing observations.
        assert starts.min() >= lo and starts.max() + 336 <= hi
        assert np.all(seconds[starts + 335] - seconds[starts] == 335 * 3600)
        assert np.all(seconds[starts + 168] - seconds[starts + 167] == 3600)
        split_info[name] = {'start_index': lo, 'end_index_exclusive': hi,
                            'start_time': str(grid[lo]), 'end_time': str(grid[hi-1]),
                            'hours': hi-lo, 'missing_hours': int(missing.iloc[lo:hi].sum()),
                            'windows': len(starts), 'excluded_windows': max(0, hi-lo-335)-len(starts)}
    tmp = DATA / 'prepared.tmp.npz'
    np.savez_compressed(tmp, **arrays)
    tmp.replace(DATA / 'prepared.npz')
    report = {'raw_rows': len(raw), 'identical_duplicate_rows_removed': duplicate_count,
              'raw_non_10min_intervals': int(times.diff().iloc[1:].ne(pd.Timedelta(minutes=10)).sum()),
              'raw_null_cells': int(raw.isna().sum().sum()), 'hourly_grid_rows': n,
              'observed_hourly_rows': int((~missing).sum()), 'missing_hours': int(missing.sum()),
              'wind_sentinel_replaced_by_zero': sentinels, 'features': list(hourly.columns),
              'target_index': list(hourly.columns).index('T (degC)'),
              'mean': mean.to_dict(), 'std_population': std.to_dict(), 'splits': split_info,
              'input_hours': INPUT_HOURS, 'horizon_hours': HORIZON,
              'normalization_observed_train_rows': len(train),
              'timezone_policy': 'Source civil timestamps; no inferred timezone or DST conversion',
              'policy': 'Remove identical duplicates; exact clock-hour samples; no gap filling; discard any window touching missing data; entire window within one chronological split',
              'csv_sha256': source['csv_sha256'],
              'prepared_sha256': hashlib.sha256((DATA / 'prepared.npz').read_bytes()).hexdigest()}
    save_json(DATA / 'preprocessing.json', report)
    print(json.dumps({k: report[k] for k in ['raw_rows', 'identical_duplicate_rows_removed', 'hourly_grid_rows', 'missing_hours', 'splits']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
