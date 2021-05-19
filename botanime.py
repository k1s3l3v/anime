import warnings

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')

from keras.models import load_model

import telebot
from PIL import ImageDraw
from PIL import ImageFont
from PIL import Image

discriminator = load_model('discriminator_15000')
generator = load_model('generator_15000')

noise = np.random.normal(0, 1, (25, 100))
gen_imgs = generator.predict(noise)

gen_imgs = 0.5 * gen_imgs + 0.5

fig, axs = plt.subplots(5, 5, figsize=(8, 8))

for i in range(5):
    for j in range(5):
        axs[i, j].imshow(gen_imgs[5 * i + j])
        axs[i, j].axis('off')

plt.show()

girl0 = Image.fromarray((gen_imgs[0] * 255).astype(np.uint8))
girl1 = Image.fromarray((gen_imgs[1] * 255).astype(np.uint8))
girl2 = Image.fromarray((gen_imgs[2] * 255).astype(np.uint8))
girl3 = Image.fromarray((gen_imgs[3] * 255).astype(np.uint8))

image0_size = girl0.size

mon_dog = Image.new('RGB', (2 * image0_size[0] + image0_size[1] // 4, 2 * image0_size[0] + image0_size[1] // 2),
                    (255, 255, 255))
mon_dog.paste(girl0, (0, 0))
mon_dog.paste(girl1, (image0_size[0] + image0_size[1] // 4, 0))
mon_dog.paste(girl2, (0, image0_size[0] + image0_size[1] // 4))
mon_dog.paste(girl3, (image0_size[0] + image0_size[1] // 4, image0_size[0] + image0_size[1] // 4))
mon_dog.save("anime1.jpeg")
mon_dog.show()

width, height = mon_dog.size

bot = telebot.TeleBot('1882671446:AAEQq6vjnmSpxs0To7AJzLLBc9BeFyX_Pq8')
userdict = []


@bot.message_handler(commands=['start'])
def start_message(message):
    sms = bot.reply_to(message, "привет, ты попал на anime fanpage, чтобы создать свой комикс введи /anime!")
    userdict.append(sms)
    bot.register_next_step_handler(sms, text1)


@bot.message_handler(commands=['anime'])
def text1(message):

    c = bot.reply_to(message, "введи первую реплику")
    userdict.append(message.text)
    bot.register_next_step_handler(c, text2)


def text2(message):
    a = bot.reply_to(message, "введи вторую реплику")
    userdict.append(message.text)
    bot.register_next_step_handler(a, text3)


def text3(message):
    b = bot.reply_to(message, "введи третью реплику")
    userdict.append(message.text)
    bot.register_next_step_handler(b, text4)


def text4(message):
    d = bot.reply_to(message, "введи четвертую реплику")
    userdict.append(message.text)
    bot.register_next_step_handler(d, sendpic)


def sendpic(message):
    userdict.append(message.text)
    draw = ImageDraw.Draw(mon_dog)
    text0 = userdict[-4]
    text1 = userdict[-3]
    text2 = userdict[-2]
    text3 = userdict[-1]

    font = ImageFont.truetype('calibri.ttf', 10)

    draw.text((3, 65), text0, (0, 0, 0), font)
    draw.text((image0_size[0] + image0_size[1] // 4 + 3, 65), text1, (0, 0, 0), font)
    draw.text((3, 2 * image0_size[0] + image0_size[1] // 4), text2, (0, 0, 0), font)
    draw.text((image0_size[0] + image0_size[1] // 4 + 3, 2 * image0_size[0] + image0_size[1] // 4), text3, (0, 0, 0),
              font)
    mon_dog.save("animeex.jpeg")
    bot.send_photo(message.chat.id, photo=open("animeex.jpeg",'rb'))

bot.polling()
