import gdown, sys


def g_downloader(file_id):
    """
    Download a file from Google Drive using its file ID.

    Params:
        - file_id (str): The file ID of the file to be downloaded from Google Drive.

    Returns:
        None
    """
    prefix = "https://drive.google.com/uc?/export=download&id="
    gdown.download(prefix + file_id)
