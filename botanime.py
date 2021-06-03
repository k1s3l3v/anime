import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
import telebot
from PIL import ImageDraw
from PIL import ImageFont
from PIL import Image
import pandas as pd

warnings.filterwarnings('ignore')
discriminator = load_model('discriminator_15000')
generator = load_model('generator_15000')

char_to_idx = {'\t': 95,
               '\n': 11,
               ' ': 0,
               '!': 28,
               '"': 52,
               '#': 79,
               '$': 94,
               '%': 83,
               '&': 87,
               "'": 22,
               '(': 71,
               ')': 70,
               '*': 81,
               '+': 93,
               ',': 24,
               '-': 37,
               '.': 14,
               '/': 78,
               '0': 59,
               '1': 60,
               '2': 63,
               '3': 65,
               '4': 68,
               '5': 66,
               '6': 73,
               '7': 69,
               '8': 72,
               '9': 74,
               ':': 62,
               ';': 84,
               '<': 90,
               '=': 85,
               '>': 10,
               '?': 30,
               '@': 92,
               'A': 35,
               'B': 41,
               'C': 43,
               'D': 40,
               'E': 46,
               'F': 54,
               'G': 50,
               'H': 36,
               'I': 27,
               'J': 55,
               'K': 49,
               'L': 45,
               'M': 38,
               'N': 39,
               'O': 42,
               'P': 48,
               'Q': 64,
               'R': 47,
               'S': 33,
               'T': 31,
               'U': 56,
               'V': 61,
               'W': 32,
               'X': 76,
               'Y': 34,
               'Z': 67,
               '[': 82,
               '\\': 58,
               ']': 77,
               '^': 96,
               '_': 86,
               'a': 4,
               'b': 26,
               'c': 19,
               'd': 15,
               'e': 1,
               'f': 21,
               'g': 18,
               'h': 8,
               'i': 5,
               'j': 44,
               'k': 25,
               'l': 12,
               'm': 17,
               'n': 6,
               'o': 3,
               'p': 23,
               'q': 57,
               'r': 9,
               's': 7,
               't': 2,
               'u': 13,
               'v': 29,
               'w': 20,
               'x': 51,
               'y': 16,
               'z': 53,
               '{': 89,
               '|': 91,
               '}': 88,
               '~': 80,
               '�': 75}
idx_to_char = {0: ' ',
               1: 'e',
               2: 't',
               3: 'o',
               4: 'a',
               5: 'i',
               6: 'n',
               7: 's',
               8: 'h',
               9: 'r',
               10: '>',
               11: '\n',
               12: 'l',
               13: 'u',
               14: '.',
               15: 'd',
               16: 'y',
               17: 'm',
               18: 'g',
               19: 'c',
               20: 'w',
               21: 'f',
               22: "'",
               23: 'p',
               24: ',',
               25: 'k',
               26: 'b',
               27: 'I',
               28: '!',
               29: 'v',
               30: '?',
               31: 'T',
               32: 'W',
               33: 'S',
               34: 'Y',
               35: 'A',
               36: 'H',
               37: '-',
               38: 'M',
               39: 'N',
               40: 'D',
               41: 'B',
               42: 'O',
               43: 'C',
               44: 'j',
               45: 'L',
               46: 'E',
               47: 'R',
               48: 'P',
               49: 'K',
               50: 'G',
               51: 'x',
               52: '"',
               53: 'z',
               54: 'F',
               55: 'J',
               56: 'U',
               57: 'q',
               58: '\\',
               59: '0',
               60: '1',
               61: 'V',
               62: ':',
               63: '2',
               64: 'Q',
               65: '3',
               66: '5',
               67: 'Z',
               68: '4',
               69: '7',
               70: ')',
               71: '(',
               72: '8',
               73: '6',
               74: '9',
               75: '�',
               76: 'X',
               77: ']',
               78: '/',
               79: '#',
               80: '~',
               81: '*',
               82: '[',
               83: '%',
               84: ';',
               85: '=',
               86: '_',
               87: '&',
               88: '}',
               89: '{',
               90: '<',
               91: '|',
               92: '@',
               93: '+',
               94: '$',
               95: '\t',
               96: '^'}


class TextRNN(nn.Module):

    def __init__(self, input_size, hidden_size, embedding_size, n_layers=1):
        super(TextRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers

        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x, hidden):
        x = self.encoder(x).squeeze(2)
        out, (ht1, ct1) = self.lstm(x, hidden)
        out = self.dropout(out)
        x = self.fc(out)
        return x, (ht1, ct1)

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device))


def evaluate(model, char_to_idx, idx_to_char, start_text=' ', prediction_len=200, temp=0.3):
    hidden = model.init_hidden()
    idx_input = [char_to_idx[char] for char in start_text]
    train = torch.LongTensor(idx_input).view(-1, 1, 1).to(device)
    predicted_text = start_text

    _, hidden = model(train, hidden)

    inp = train[-1].view(-1, 1, 1)

    for i in range(prediction_len):
        output, hidden = model(inp.to(device), hidden)
        output_logits = output.cpu().data.view(-1)
        p_next = F.softmax(output_logits / temp, dim=-1).detach().cpu().data.numpy()
        top_index = np.random.choice(len(char_to_idx), p=p_next)
        inp = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)
        predicted_char = idx_to_char[top_index]
        predicted_text += predicted_char

    return predicted_text


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = TextRNN(input_size=len(idx_to_char), hidden_size=128, embedding_size=128, n_layers=2)
model.to(device)

model.load_state_dict(torch.load('text_gen'))
model.eval()


def get_text(length=80, start=". "):
    text = start

    while text == start:
        text = (evaluate(
            model,
            char_to_idx,
            idx_to_char,
            temp=0.3,
            prediction_len=length,
            start_text=text
        )
        )
    if start == ". ":
        text = text[len(start):]
    return text


def gen_images():
    noise = np.random.normal(0, 1, (25, 100))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5

    return gen_imgs


fig, axs = plt.subplots(5, 5, figsize=(8, 8))

images = gen_images()
index = 0
generate = False
for i in range(5):
    for j in range(5):
        axs[i, j].imshow(images[5 * i + j])
        axs[i, j].axis('off')

plt.show()
nRows = 1000
df1 = pd.read_csv('anime.csv', delimiter=',', nrows=nRows)
df1.dataframeName = 'anime.csv'
nRow, nCol = df1.shape

genreslist = list(df1['genre'])

dic = {}
for i in range(len(df1['genre'])):
    for g in (df1['genre'][i]).split(', '):
        if g in dic.keys():
            dic[g].append(df1['name'][i])
        else:
            dic[g] = [df1['name'][i]]

bot = telebot.TeleBot('1882671446:AAEQq6vjnmSpxs0To7AJzLLBc9BeFyX_Pq8')
user_dict = []
keyboard = telebot.types.ReplyKeyboardMarkup(True, True)
keyboard.row("/generatemycomics", "/generatetext", "/generatecomics", "/rec")


@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, "Здравствуйте, Вы попали на anime fanpage!\nЧтобы создать свой комикс,"
                                      " введите /generatemycomics\n"
                                      "Чтобы посмотреть на случайный комикс, введите /generatecomics\n"
                                      "Чтобы посмотреть на случайно составленный текст,"
                                      " основанный на аниме субтитрах, введите /generatetext\n"
                                      "Чтобы посмотреть подборку, введите /rec", reply_markup=keyboard)


@bot.message_handler(commands=['generatecomics'])
def gen_message(message):
    global generate
    generate = True
    gen = bot.reply_to(message, "Введите любую букву")

    bot.register_next_step_handler(gen, send_pic)


@bot.message_handler(commands=['generatetext'])
def gen_message(message):
    bot.reply_to(message, " ".join(get_text(length=400).split()[:-1]))


@bot.message_handler(commands=['generatemycomics'])
def text1(message):
    global generate
    generate = False
    c = bot.reply_to(message, "Введите первую реплику")
    user_dict.append(message.text)
    bot.register_next_step_handler(c, text2)


def text2(message):
    a = bot.reply_to(message, "Введите вторую реплику")
    user_dict.append(message.text)
    bot.register_next_step_handler(a, text3)


def text3(message):
    b = bot.reply_to(message, "Введите третью реплику")
    user_dict.append(message.text)
    bot.register_next_step_handler(b, text4)


def text4(message):
    d = bot.reply_to(message, "Введите четвертую реплику")
    user_dict.append(message.text)
    bot.register_next_step_handler(d, send_pic)


def send_pic(message):
    global images
    global index
    global generate
    if index > 23:
        images = gen_images()
        index = 0

    girl0 = Image.fromarray((images[index] * 255).astype(np.uint8))
    girl1 = Image.fromarray((images[index + 1] * 255).astype(np.uint8))
    girl2 = Image.fromarray((images[index + 2] * 255).astype(np.uint8))
    girl3 = Image.fromarray((images[index + 3] * 255).astype(np.uint8))

    image0_size = girl0.size

    mon_dog = Image.new('RGB', (2 * image0_size[0] + image0_size[1] // 4, 2 * image0_size[0] + image0_size[1] // 2),
                        (255, 255, 255))
    mon_dog.paste(girl0, (0, 0))
    mon_dog.paste(girl1, (image0_size[0] + image0_size[1] // 4, 0))
    mon_dog.paste(girl2, (0, image0_size[0] + image0_size[1] // 4))
    mon_dog.paste(girl3, (image0_size[0] + image0_size[1] // 4, image0_size[0] + image0_size[1] // 4))
    mon_dog.save("anime1.jpeg")
    mon_dog.show()

    user_dict.append(message.text)
    draw = ImageDraw.Draw(mon_dog)
    if generate:
        txt = get_text().split()[:-1]
        print(txt)
        print("Sock")
        spl = len(txt) // 8 + 1
        txt0 = " ".join(txt[:spl])
        txt1 = " ".join(txt[spl:2 * spl])
        txt2 = " ".join(txt[2 * spl:3 * spl])
        txt3 = " ".join(txt[3 * spl:])
    else:
        txt0 = user_dict[-4]
        txt1 = user_dict[-3]
        txt2 = user_dict[-2]
        txt3 = user_dict[-1]

    font = ImageFont.truetype('calibri.ttf', 10)

    draw.text((3, 65), txt0, (0, 0, 0), font)
    draw.text((image0_size[0] + image0_size[1] // 4 + 3, 65), txt1, (0, 0, 0), font)
    draw.text((3, 2 * image0_size[0] + image0_size[1] // 4), txt2, (0, 0, 0), font)
    draw.text((image0_size[0] + image0_size[1] // 4 + 3, 2 * image0_size[0] + image0_size[1] // 4), txt3, (0, 0, 0),
              font)
    mon_dog.save("animeex.jpeg")
    bot.send_photo(message.chat.id, photo=open("animeex.jpeg", 'rb'))
    index += 4


genres = []


def get_list_genres(genres):
    result = dic[genres[0]]
    for genre in genres[1:]:
        result = list(set(result) & set(dic[genre]))
    return result


@bot.message_handler(commands=['rec'])
def genre1(message):
    gen1 = bot.reply_to(message, f"Введите жанр\nСписок жанров: {', '.join(list(dic.keys()))}")
    bot.register_next_step_handler(gen1, genre2)


def genre2(message):
    if message.text == '/ready':
        result = get_list_genres(genres)
        if len(result) == 0:
            bot.send_message(message.chat.id, "Таких аниме не было найдено")
        else:
            bot.send_message(message.chat.id, f"Вот ваш список аниме: {', '.join(result[:10]) if len(result) > 10 else ', '.join(result)}")
        genres.clear()
    elif message.text not in dic.keys():
        gen2 = bot.reply_to(message, f"Такого жанра не существует, вот список доступных жанров: {', '.join(list(dic.keys()))}\n"
                                     f"Попробуйте снова")
        bot.register_next_step_handler(gen2, genre2)
    else:
        keyboard = telebot.types.ReplyKeyboardMarkup(True, True)
        keyboard.row("/ready")
        gen2 = bot.reply_to(message, "Введите следующий жанр или нажмите /ready:", reply_markup=keyboard)
        genres.append(message.text)
        bot.register_next_step_handler(gen2, genre2)


bot.polling()


