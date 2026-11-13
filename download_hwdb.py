"""下载 CASIA 官方 GNT 压缩包，保存本地 SHA-256 和来源信息。"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'data' / 'hwdb_official'
BASE = 'https://nlpr.ia.ac.cn/databases/Download/Offline/CharData/'
FILES = {'Gnt1.1TrainPart1.zip': 919543940,
         'Gnt1.1TrainPart2.zip': 966309492,
         'Gnt1.1Test.zip': 479659120}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def ranged_download(name, expected):
    """三段并发；共三个包，最多九条连接。已有连续前缀与完整分段可复用。"""
    temporary = DEST / (name + '.part')
    parts_dir = DEST / (name + '.parts')
    parts_dir.mkdir(exist_ok=True)
    offset = temporary.stat().st_size if temporary.exists() else 0
    if offset > expected:
        raise ValueError('Partial file larger than expected archive')
    chunk_size = 32 * 1024 * 1024
    ranges = [(start, min(start + chunk_size, expected) - 1)
              for start in range(offset, expected, chunk_size)]

    def part(bounds):
        start, end = bounds
        target = parts_dir / f'{start:012d}-{end:012d}.bin'
        size = end - start + 1
        if target.exists() and target.stat().st_size == size:
            return target
        for attempt in range(3):
            try:
                request = Request(BASE + name, headers={'Range': f'bytes={start}-{end}'})
                with urlopen(request, timeout=45) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{expected}':
                        raise ValueError('Server did not honor requested byte range')
                    with target.open('wb') as f:
                        while chunk := response.read(1024 * 1024):
                            f.write(chunk)
                if target.stat().st_size != size:
                    raise ValueError('Incomplete range')
                return target
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    with ThreadPoolExecutor(max_workers=3) as pool:
        # 顺序写回连续前缀，因此中断后仍能安全恢复。
        with temporary.open('ab') as output:
            for target in pool.map(part, ranges):
                with target.open('rb') as source:
                    for chunk in iter(lambda: source.read(4 * 1024 * 1024), b''):
                        output.write(chunk)
                output.flush()
                print(f'{name}: {output.tell() / expected:.1%} ({output.tell() / 1e6:.0f} MB)', flush=True)
    if temporary.stat().st_size != expected:
        raise ValueError('Completed archive length mismatch')
    temporary.replace(DEST / name)


def download(name):
    path = DEST / name
    expected = FILES[name]
    headers = {}
    if not path.exists():
        print(f'DOWNLOAD {name}, resuming with HTTP ranges', flush=True)
        ranged_download(name, expected)
    if path.stat().st_size != expected:
        raise ValueError(f'Existing file has unexpected size: {path}')
    print(f'CHECKING {name}', flush=True)
    digest = sha256_file(path)
    with zipfile.ZipFile(path) as archive:
        # 对所有成员解压验证 CRC，不仅检查 ZIP 文件头。
        bad = archive.testzip()
        if bad:
            raise ValueError(f'ZIP CRC error: {bad}')
        members = archive.namelist()
    result = {'file': name, 'url': BASE + name, 'bytes': expected, 'sha256': digest,
              'sha256_origin': 'locally computed; no publisher SHA-256 supplied',
              'zip_crc_verified': True, 'member_count': len(members),
              'gnt_members': [x for x in members if x.lower().endswith('.gnt')],
              'response_headers': headers,
              'verified_at_utc': datetime.now(timezone.utc).isoformat()}
    (DEST / (name + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'VERIFIED {name}: SHA256={digest}; {len(result["gnt_members"])} GNT files', flush=True)
    return result


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        manifests = list(pool.map(download, FILES))
    (DEST / 'sources.json').write_text(json.dumps(manifests, ensure_ascii=False, indent=2), encoding='utf-8')
    print('ALL_OFFICIAL_ARCHIVES_VERIFIED', flush=True)


if __name__ == '__main__':
    main()
