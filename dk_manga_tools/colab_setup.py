import glob
from google.colab import drive
import time
import colab_env
from colab_env import envvar_handler
import os
from marvin import config


def mount_drive():
    drive.mount("/content/gdrive", force_remount = True)


def check_mount_status():
    files = []
    start_time = time.time()
    while len(files) == 0:
        print("Checking for mount completion...")
        files = glob.glob("/content/gdrive/My Drive/sas/mangawork/manga/sandbox/galaxyzoo3d/v3_0_0/*.fits.gz")
        time.sleep(30.)
    end_time = time.time()
    print(f"Mount complete! after {(end_time-start_time):.1f} seconds")

def add_sas_path():
    envvar_handler.add_env("SAS_BASE_DIR", "/content/gdrive/My Drive/sas", overwrite=True)
    colab_env.RELOAD()
    print("Current SAS_BASE_DIR set to:")
    os.system("echo $SAS_BASE_DIR")

def make_netrc():
    pw = input("Enter SDSS Login Password: ")

    os.system("echo 'machine api.sdss.org' >> $HOME/.netrc")
    os.system("echo '   login sdss' >> $HOME/.netrc")
    os.system(f"echo '   password {pw}' >> $HOME/.netrc")

    os.system("echo 'machine data.sdss.org' >> $HOME/.netrc")
    os.system("echo '   login sdss' >> $HOME/.netrc")
    os.system(f"echo '   password {pw}' >> $HOME/.netrc")

    os.system("chmod 600 $HOME/.netrc")

def prepare_marvin():
    from marvin import config
    config.access = "collab"
    config.mode = "auto"
    config.setRelease("MPL-10")
    config.login()

def setup(public = True):
    print("Mounting Google Drive")
    mount_drive()
    print("Checking Google Drive mount status (this can take ~5 minutes)")
    check_mount_status()
    print("Setting up SAS_BASE_DIR")
    add_sas_path()
    if not public:
        print("Setting up SDSS login")
        print("Note: you will need the SDSS login password for data access")
        make_netrc()
        print("Setting up Marvin for MPL-10")
        prepare_marvin()
    print("Ready to use!")



