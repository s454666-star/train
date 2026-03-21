import asyncio
import os
import traceback
from telethon import TelegramClient
from telethon.tl.functions.messages import SendMediaRequest
from telethon.tl.types import InputMediaUploadedDocument, DocumentAttributeVideo, DocumentAttributeFilename

api_id = 27946982
api_hash = '2324efd7bed05b02a63e3809fa93048c'
file_path = r'Z:\video(重跑)\自拍_1396.mp4'

async def main():
    client = TelegramClient('session/main_account', api_id, api_hash)
    await client.start()
    print('started', flush=True)
    peer = await client.get_input_entity('filestoebot')
    print('peer', flush=True)
    uploaded = await client.upload_file(file_path, file_name=os.path.basename(file_path))
    print('uploaded', flush=True)
    media = InputMediaUploadedDocument(
        file=uploaded,
        mime_type='video/mp4',
        attributes=[
            DocumentAttributeVideo(duration=8.2, w=1440, h=2560, supports_streaming=True),
            DocumentAttributeFilename(file_name=os.path.basename(file_path)),
        ],
        force_file=False,
        nosound_video=False,
    )
    print('media', flush=True)
    result = await client(SendMediaRequest(peer=peer, media=media, message=''))
    print(type(result).__name__, flush=True)
    print('ok', flush=True)
    await client.disconnect()

try:
    asyncio.run(main())
except Exception:
    traceback.print_exc()
    raise
