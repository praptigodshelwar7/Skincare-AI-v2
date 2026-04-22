import ssl
import json
import urllib.request
import os
import zipfile
import sys

def download_and_extract(package_name, dest_dir="venv/Lib/site-packages"):
    print(f"Fetching metadata for {package_name}...")
    url = f"https://pypi.org/pypi/{package_name}/json"
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, context=ctx) as response:
            data = json.loads(response.read().decode('utf-8'))
            urls = data['urls']
            wheel_url = None
            wheel_filename = None
            
            # Prefer wheel for cp312 and win_amd64 if available, otherwise just any valid wheel
            for release in urls:
                if release['packagetype'] == 'bdist_wheel':
                    fn = release['filename']
                    if 'win_amd64' in fn and 'cp312' in fn:
                        wheel_url = release['url']
                        wheel_filename = release['filename']
                        break
            
            # Fallback for any compatible wheel
            if not wheel_url:
                for release in urls:
                    if release['packagetype'] == 'bdist_wheel':
                        fn = release['filename']
                        if 'any' in fn or ('cp312' in fn and 'win' in fn):
                            wheel_url = release['url']
                            wheel_filename = fn
                            break

            if not wheel_url:
                print(f"Could not find a valid wheel for {package_name} on Windows Python 3.12")
                return False

            if not os.path.exists(wheel_filename):
                print(f"Downloading {wheel_filename}...")
                urllib.request.urlretrieve(wheel_url, wheel_filename)
            else:
                print(f"Already downloaded {wheel_filename}")
                
            print(f"Extracting {wheel_filename} to {dest_dir}...")
            # create dir if doesn't exist
            os.makedirs(dest_dir, exist_ok=True)
            with zipfile.ZipFile(wheel_filename, 'r') as zip_ref:
                zip_ref.extractall(dest_dir)
            
            print(f"Successfully extracted {package_name}!")
            return True
    except Exception as e:
        print(f"Error fetching/extracting {package_name}: {e}")
        return False

# Extract into venv
dest = os.path.join("venv", "Lib", "site-packages")
download_and_extract("onnxruntime", dest)
download_and_extract("rapidocr-onnxruntime", dest)
