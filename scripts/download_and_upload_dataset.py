import os
import sys
import subprocess
from google.cloud import storage

def log(msg):
    print(f"[MAFT DATA] {msg}")

# Download CMU-MOSEI using the CMU Multimodal SDK
def download_mosei(output_dir):
    log("Installing cmumosei SDK if needed...")
    #subprocess.run([sys.executable, "-m", "pip", "install", "mmsdk"], check=True)
    from mmsdk import mmdatasdk
    log("Downloading CMU-MOSEI dataset...")
    mmdatasdk.mmdataset(mmdatasdk.cmu_mosei.labels, root=output_dir)
    log("Download complete.")

def upload_to_gcs(local_dir, bucket_name, gcs_path_prefix="data/"):
    log(f"Uploading {local_dir} to gs://{bucket_name}/{gcs_path_prefix}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            rel_path = os.path.relpath(local_path, local_dir)
            blob = bucket.blob(os.path.join(gcs_path_prefix, rel_path))
            blob.upload_from_filename(local_path)
            log(f"Uploaded {rel_path}")
    log("Upload to GCS complete.")

def main():
    output_dir = "/opt/maft/data"
    bucket_name = os.environ.get("MAFT_GCS_BUCKET")
    if not bucket_name:
        print("Set MAFT_GCS_BUCKET environment variable.")
        sys.exit(1)
    download_mosei(output_dir)
    upload_to_gcs(output_dir, bucket_name)

if __name__ == "__main__":
    main() 