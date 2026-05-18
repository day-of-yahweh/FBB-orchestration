import os
import traceback
import asyncio
from telegram import Bot


async def send_message(text, chat_id, token):
    bot = Bot(token=token)
    await bot.send_message(chat_id=chat_id,
                           text=text)

async def get_chat_ids(username, token):
    bot = Bot(token=token)
    updates = await bot.get_updates()
    for update in updates:
        chat = update.effective_chat
        user = update.effective_user
        if chat is None or user is None:
            continue
        if user.username == username:
            return chat.id

def notify(text, uname='dkhlebn'):
    BOT_TOKEN = os.environ["TGBOT_TOKEN"]
    chat_id = asyncio.run(get_chat_ids(uname, BOT_TOKEN))
    asyncio.run(send_message(text, chat_id, BOT_TOKEN))

def mainloop():
    print("Doing important work...")
    x = 1 / 2


if __name__ == "__main__":
    try:
        mainloop()
    except Exception:
        error = traceback.format_exc()
        msg = "*Script failed*\n\n```" + error + "```"
        notify(msg)
    else:
        msg = notify("*Script finished successfully*")
