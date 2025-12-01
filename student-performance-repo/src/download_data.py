#!/usr/bin/env python3
"""Small helper to download the UCI Student Performance dataset (mathematics).
Note: Internet access required. If not available, download manually:
https://archive.ics.uci.edu/ml/datasets/Student+Performance
"""
import argparse
import sys
try:
    import requests
except Exception:
    print('requests is required to download automatically. Install with: pip install requests')
    sys.exit(1)

def download(out_path):
    # direct file URL (mirror may change). This tries the common UCI link for student-mat.csv in CSV format.
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip'
    print('Downloading', url)
    r = requests.get(url)
    r.raise_for_status()
    import zipfile, io
    z = zipfile.ZipFile(io.BytesIO(r.content))
    # student-mat.csv inside zip
    name = 'student-mat.csv'
    if name not in z.namelist():
        # try with folder
        for n in z.namelist():
            if n.endswith(name):
                name = n
                break
    print('Extracting', name)
    with z.open(name) as src, open(out_path, 'wb') as dst:
        dst.write(src.read())
    print('Saved to', out_path)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', default='data/student-mat.csv', help='Output path for CSV')
    args = p.parse_args()
    download(args.out)
