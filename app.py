import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import random
import pathlib
import requests
from rag import index_slack_file_bytes, answer_query, delete_all_embeddings


# This sample slack application uses SocketMode
# For the companion getting started setup guide,
# see: https://docs.slack.dev/tools/bolt-python/getting-started

# Initializes your app with your bot token
BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]   # xoxb-...

app = App(token=BOT_TOKEN)
SAVE_DIR = pathlib.Path("./saved_files")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {"Authorization": f"Bearer {BOT_TOKEN}"}

INDEXED_FILE_IDS = set()


# Listens to incoming messages that contain "hello"
@app.message("hello")
def message_hello(message, say):
    # say() sends a message to the channel where the event was triggered
    say(
        blocks=[
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"Hey there <@{message['user']}>!"},
                "accessory": {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Click Me"},
                    "action_id": "button_click",
                },
            }
        ],
        text=f"Hey there <@{message['user']}>!",
    )


@app.action("button_click")
def action_button_click(body, ack, say):
    # Acknowledge the action
    ack()
    say(f"<@{body['user']['id']}> clicked the button")


@app.message("goodbye")
def message_goodbye(say):
    responses = ["Adios", "Au revoir", "Farewell"]
    parting = random.choice(responses)
    say(f"{parting}!")

def download_slack_file(url_private_download: str) -> bytes:
    # 1) hit Slack URL without auto-redirects
    r = requests.get(url_private_download, headers=HEADERS, allow_redirects=False, timeout=60)

    # 2) If Slack redirects to workspace webapp ?redir=/files-pri/..., convert to files.slack.com path
    if r.status_code in (301, 302, 303, 307, 308) and "Location" in r.headers:
        loc = r.headers["Location"]        
        r = requests.get(loc, headers=HEADERS, timeout=60)

    r.raise_for_status()

    # 3) refuse HTML (means you didn’t get file bytes)
    ct = (r.headers.get("Content-Type") or "").lower()
    if "text/html" in ct:
        raise RuntimeError(f"Got HTML, not file bytes. URL={r.url}")

    return r.content

BOT_USER_ID = None

@app.event("app_home_opened")
def _cache_bot_id(event, client, logger):
    global BOT_USER_ID
    if BOT_USER_ID:
        return
    BOT_USER_ID = client.auth_test()["user_id"]
    logger.info(f"Cached BOT_USER_ID={BOT_USER_ID}")

@app.event("message")
def on_message(event, client, logger):
    global BOT_USER_ID

    # Ignore bot messages
    if event.get("bot_id") or event.get("subtype") == "bot_message":
        return

    # Ensure bot id cached (works even if app_home_opened didn't happen)
    if not BOT_USER_ID:
        BOT_USER_ID = client.auth_test()["user_id"]

    channel = event.get("channel")
    thread_ts = event.get("thread_ts") or event.get("ts")

    subtype = event.get("subtype")

    text = (event.get("text") or "").strip().lower()

    # Only trigger if user mentions the bot AND types delete
    if f"<@{BOT_USER_ID}>" in (event.get("text") or "") and text.endswith("delete"):
        remaining = delete_all_embeddings()
        client.chat_postMessage(
            channel=event["channel"],
            text=f"✅ Deleted all embeddings."
        )
        return

     # ---- A) File upload case (subtype=file_share) ----
    if subtype == "file_share":
        for f in event.get("files", []):
            file_id = f.get("id")
            if not file_id:
                continue

            if file_id in INDEXED_FILE_IDS:
                continue

            info = client.files_info(file=file_id)
            file_obj = info["file"]

            url = file_obj.get("url_private_download") or file_obj.get("url_private")
            name = file_obj.get("name") or file_obj.get("title") or f"{file_id}.bin"
            dest = SAVE_DIR / f"{file_id}-{name}"

            if not url:
                client.chat_postMessage(
                    channel=channel,
                    thread_ts=thread_ts,
                    text=f"❌ Could not find a download URL for `{name}`.",
                )
                continue

            try:
                data = download_slack_file(url)
                dest.write_bytes(data)

                user_id = event['user']
                result = client.users_info(user=user_id)
                user_name = result['user']['profile']['display_name'] or result['user']['real_name']
    

                # Index into RAG
                ids = index_slack_file_bytes(file_bytes=data, file_obj=file_obj, user_id=user_name, channel_id=channel)

                INDEXED_FILE_IDS.add(file_id)

                client.chat_postMessage(
                    channel=channel,
                    thread_ts=thread_ts,
                    text=f"✅ Saved `{name}` and indexed {len(ids)} chunks.",
                )
            except Exception as e:
                logger.exception("Failed processing uploaded file")
                client.chat_postMessage(
                    channel=channel,
                    thread_ts=thread_ts,
                    text=f"❌ Failed to process `{name}`: {e}",
                )
            return
    
    
    if (event.get("type") == "message" and not event.get("subtype") and bool(event.get("text"))):
        text = (event.get("text") or "").strip()

        
        if not text:
            return

        try:
            answer = answer_query(text, slack_channel=channel, k=4)
            client.chat_postMessage(channel=channel, thread_ts=thread_ts, text=answer)
        except Exception as e:
            logger.exception("Failed answering query")
            client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=f"❌ Error answering: {e}",
        )

    



# Start your app
if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
