# Modified from https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive

# To download houses weights run
# python Download_Weights.py 1aPCwYXFAOmklmNMLMh81Yduw5UrbHqkN Houses/trained_weights_final.h5

# To download openeings weights run
# python Download_Weights.py 1FbvHzQWCjucXPbTbI4S1MnBLkAi58Mxv Openings/trained_weights_final.h5

import requests
import os
import progressbar


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith("download_warning"):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
            i = 0
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    bar.update(i)
                    i += 1
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python google_drive.py drive_file_id destination_file_path")
    else:
        # TAKE ID FROM SHAREABLE LINK
        file_id = sys.argv[1]
        # DESTINATION FILE ON YOUR DISK
        destination = os.path.join(os.getcwd(), sys.argv[2])
        download_file_from_google_drive(file_id, destination)
