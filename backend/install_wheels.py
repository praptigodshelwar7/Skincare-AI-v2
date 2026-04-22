import ssl
import json
import urllib.request
import os
import subprocess
import sys

def download_and_install(package_name):
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

            print(f"Downloading {wheel_filename}...")
            urllib.request.urlretrieve(wheel_url, wheel_filename)
            
            print(f"Installing {wheel_filename} using offline pip...")
            python_exe = sys.executable
            # Try to install silently without index
            res = subprocess.run([python_exe, "-m", "pip", "install", wheel_filename, "--no-index", "--find-links=."], capture_output=True, text=True)
            if res.returncode == 0:
                print(f"Success installing {package_name}")
                return True
            else:
                print(f"Failed to install {package_name}:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")
                return False
    except Exception as e:
        print(f"Error fetching {package_name}: {e}")
        return False

download_and_install("onnxruntime")
download_and_install("rapidocr-onnxruntime")
