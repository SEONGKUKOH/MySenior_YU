#pip install --upgrade google-api-python-client
#pip install oauth2client
# 전송할 구글 드라이브 폴더 : 1-4kEumtHS1LwKgFpa2K6Tu4EOtAPvbk1

# mediapipe를 통해 캡쳐한 사진을 구글 드라이드로 전송한다.

import datetime
import os

from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools


file_list = os.listdir("C:/Users/SEONGKUK/pose_dcgan/Emergency")
file_list_py = [file for file in file_list if file.endswith(".png")]
latest_picture = file_list_py[-1]



try :
    import argparse
    flags = argparse.ArgumentParser(parents=[tools.argparser]).parse_args()
except ImportError:
    flags = None

SCOPES = 'https://www.googleapis.com/auth/drive.file'
store = file.Storage('storage.json')
creds = store.get()

if not creds or creds.invalid:
    print("make new storage data file ")
    flow = client.flow_from_clientsecrets('C:/Users/SEONGKUK/pose_dcgan/client_secret_drive.json', SCOPES)
    creds = tools.run_flow(flow, store, flags) if flags else tools.run(flow, store)

DRIVE = build('drive', 'v3', http=creds.authorize(Http()))

FILES = (
    ('C:/Users/SEONGKUK/pose_dcgan/Emergency/'+latest_picture),
)


# FILES = (
#     ('C:/Users/SEONGKUK/pose_dcgan/Emergency/'+latest_picture),
# )

folder_id = '1-4kEumtHS1LwKgFpa2K6Tu4EOtAPvbk1'

for file_title in FILES :
    file_name = file_title
    
    suffix = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    fileName = suffix + '.jpg'
    metadata = {'name': fileName,
                'parents' : [folder_id],
                'mimeType': None
                }

    res = DRIVE.files().create(body=metadata, media_body=file_name).execute()
    if res:
        print('Uploaded "%s" (%s)' % (file_name, res['mimeType']))




print("google drive 실행됨")