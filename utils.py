import glob


def get_latest_version(root_dir: str):
    version = 1
    directories = glob.glob(root_dir + "/*/")
    for directory in directories:
        last_version = int(directory.split("/")[-2].split("_")[-1])
        if last_version == version:
            version = last_version + 1

    return version
