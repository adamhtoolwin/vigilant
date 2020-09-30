import glob


def get_latest_version(root_dir: str):
    last_version = 1
    directories = glob.glob(root_dir + "/*/")
    for directory in directories:
        version = int(directory.split("/")[-2].split("_")[-1])
        if version >= last_version:
            last_version = version + 1

    return last_version
