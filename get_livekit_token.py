import os
from livekit import api
from dotenv import load_dotenv

load_dotenv()

ROOM_NAME = "my-test-room"  
PARTICIPANT_NAME = "TestUser"

token = api.AccessToken(
    os.getenv('LIVEKIT_API_KEY'),
    os.getenv('LIVEKIT_API_SECRET')
).with_identity("identity-1") \
 .with_name(PARTICIPANT_NAME) \
 .with_grants(api.VideoGrants(
    room_join=True,
    room=ROOM_NAME,
 ))

print(f"\nToken:\n")
print(token.to_jwt())
