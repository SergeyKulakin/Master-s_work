# -*- coding: utf-8 -*-
import telebot
from telebot import types
from telebot.types import InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery
import models
from models import gen_data, plot_report
from base import check_num
import os

import warnings
warnings.filterwarnings("ignore")

with open('TOKEN.txt', 'r') as f:
    token = f.readline()
bot = telebot.TeleBot(token)

def build_menu(buttons, n_cols,
               header_buttons=None,
               footer_buttons=None):
    '''Функция создания встроенного стартового меню'''
    menu = [buttons[i:i + n_cols] for i in range(0, len(buttons), n_cols)]
    if header_buttons:
        menu.insert(0, [header_buttons])
    if footer_buttons:
        menu.append([footer_buttons])
    return menu
###
###
###
###
###
###
@bot.message_handler(commands=['start'])
@bot.callback_query_handler(func=lambda c: c.data == 'back')
def welcome_start(msg, text='Привет, я Fraud Detector! \n'
                            'Выберите что вас интересует из предложенного:'):
    '''Функция Старт'''
    button_list = [InlineKeyboardButton('Описание моделей машинного обучения', callback_data='desc'),
                   InlineKeyboardButton('Тестирование моделей машинного обучения', callback_data='examp')]


    # сборка клавиатуры из кнопок `InlineKeyboardButton`
    reply_markup = InlineKeyboardMarkup(build_menu(button_list, n_cols=1))
    # отправка клавиатуры в чат
    bot.send_message(msg.from_user.id, text=text, reply_markup=reply_markup)
###
###
###
###
###
###
@bot.callback_query_handler(func=lambda c: c.data == 'desc')
def button_desc(callback_query: types.CallbackQuery):
    '''Функция определения ответа на кнопку Описание'''
    bot.answer_callback_query(callback_query.id)
    bot.send_message(callback_query.from_user.id, 'Базовыми моделями являются: \n'
                                                  '- LGBM; \n'
                                                  '- Stacking Classifier.')
    bot.send_message(callback_query.from_user.id, 'Архитектура предложенной модели "Custom" представлена на'
                                                  ' рисунке ниже')
    with open('Schem.png', 'rb') as schem:
        bot.send_photo(callback_query.from_user.id, schem,  protect_content=True)

    '''КНОПКА НАЗАД'''
    button_list = [InlineKeyboardButton('Назад', callback_data='back')]

    # сборка клавиатуры из кнопок `InlineKeyboardButton`
    reply_markup = InlineKeyboardMarkup(build_menu(button_list, n_cols=1))
    # отправка клавиатуры в чат
    bot.send_message(callback_query.from_user.id, text='Вернуться в главное меню',
                     reply_markup=reply_markup)
###
###
###
###
###
###
@bot.callback_query_handler(func=lambda c: c.data == 'examp')
def button_examp(msg, text='В этом разделе вы можете протестировать модели'
                           ' на различных данных, которые содержат'
                           ' мошеннические транзакции'):
    '''Функция определения ответа на кнопку Пример'''
    button_list = [InlineKeyboardButton('Реальные данные', callback_data='real'),
                   InlineKeyboardButton('Синтетические данные', callback_data='gan'),
                   InlineKeyboardButton('Назад', callback_data='back')]

    # сборка клавиатуры из кнопок `InlineKeyboardButton`
    reply_markup = InlineKeyboardMarkup(build_menu(button_list, n_cols=1))
    # отправка клавиатуры в чат
    bot.send_message(msg.from_user.id, text=text, reply_markup=reply_markup)
###
###
###
###
###
###
@bot.callback_query_handler(func=lambda c: c.data == 'real')
def button_examp(msg, text='Выберите модель для проверки на реальных данных'):
    '''Функция определения ответа на кнопку Пример'''
    button_list = [InlineKeyboardButton('LGBM', callback_data='r_lgbm'),
                   InlineKeyboardButton('Stacking Classifier', callback_data='r_sk'),
                   InlineKeyboardButton('Custom model', callback_data='r_custom')]
    '''КНОПКА НАЗАД'''
    button_list_2 = [InlineKeyboardButton('В меню', callback_data='back'),
                     InlineKeyboardButton('Шаг назад', callback_data='examp')]

        # сборка клавиатуры из кнопок `InlineKeyboardButton`
    reply_markup = InlineKeyboardMarkup(build_menu(button_list, n_cols=3))
    # сборка клавиатуры из кнопок `InlineKeyboardButton`
    reply_markup_2 = InlineKeyboardMarkup(build_menu(button_list_2, n_cols=2))
    # отправка клавиатуры в чат
    bot.send_message(msg.from_user.id, text=text, reply_markup=reply_markup)
    bot.send_message(msg.from_user.id, text='Вернуться в меню или на шаг назад', reply_markup=reply_markup_2)
#===========================================#
#===========================================#
@bot.callback_query_handler(func=lambda c: c.data == 'r_lgbm')
def button_examp(msg):
    '''Функция определения ответа на кнопку LGBM REAL'''
    model = 'LGBM'
    tmp = models.predict_real(model)
    bot.send_message(msg.from_user.id, text='Результаты LGBM:')
    bot.send_message(msg.from_user.id, text=f'{tmp}')
    bot.send_message(msg.from_user.id, text='Результаты получены на публичном '
                                            'датасете: https://www.kaggle.com/'
                                            'datasets/mlg-ulb/creditcardfraud?'
                                            'resource=download')
    '''КНОПКА НАЗАД'''
    button_list = [InlineKeyboardButton('В меню', callback_data='back'),
                   InlineKeyboardButton('Шаг назад', callback_data='real')]

    # сборка клавиатуры из кнопок `InlineKeyboardButton`
    reply_markup = InlineKeyboardMarkup(build_menu(button_list, n_cols=2))
    # отправка клавиатуры в чат
    bot.send_message(msg.from_user.id, text='Вернуться в меню или на шаг назад',
                     reply_markup=reply_markup)
#===========================================#
#===========================================#
@bot.callback_query_handler(func=lambda c: c.data == 'r_sk')
def button_examp(msg):
    '''Функция определения ответа на кнопку SK REAL'''
    model = 'SK'
    tmp = models.predict_real(model)
    bot.send_message(msg.from_user.id, text='Результаты StackingClassifier:')
    bot.send_message(msg.from_user.id, text=f'{tmp}')
    bot.send_message(msg.from_user.id, text='Результаты получены на публичном '
                                            'датасете: https://www.kaggle.com/'
                                            'datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022?'
                                            'resource=download')
    '''КНОПКА НАЗАД'''
    button_list = [InlineKeyboardButton('В меню', callback_data='back'),
                   InlineKeyboardButton('Шаг назад', callback_data='real')]

    # сборка клавиатуры из кнопок `InlineKeyboardButton`
    reply_markup = InlineKeyboardMarkup(build_menu(button_list, n_cols=2))
    # отправка клавиатуры в чат
    bot.send_message(msg.from_user.id, text='Вернуться в меню или на шаг назад',
                     reply_markup=reply_markup)
#===========================================#
#===========================================#
@bot.callback_query_handler(func=lambda c: c.data == 'r_custom')
def button_examp(msg):
    '''Функция определения ответа на кнопку Custom REAL'''

    model = 'Custom'
    tmp = models.predict_real(model)
    bot.send_message(msg.from_user.id, text='Результаты Custom:')
    bot.send_message(msg.from_user.id, text=f'{tmp}')
    bot.send_message(msg.from_user.id, text='Результаты получены на публичном '
                                            'датасете: https://www.kaggle.com/'
                                            'datasets/mlg-ulb/creditcardfraud?'
                                            'resource=download')
    '''КНОПКА НАЗАД'''
    button_list = [InlineKeyboardButton('В меню', callback_data='back'),
                   InlineKeyboardButton('Шаг назад', callback_data='real')]

    # сборка клавиатуры из кнопок `InlineKeyboardButton`
    reply_markup = InlineKeyboardMarkup(build_menu(button_list, n_cols=2))
    # отправка клавиатуры в чат
    bot.send_message(msg.from_user.id, text='Вернуться в меню или на шаг назад',
                     reply_markup=reply_markup)
###
###
###
###
###
###
@bot.callback_query_handler(func=lambda c: c.data == 'gan')
def button_examp(msg, text='Выберите модель для проверки на синтетических данных'):
    '''Функция определения ответа на кнопку Пример'''
    button_list = [InlineKeyboardButton('LGBM', callback_data='g_lgbm'),
                   InlineKeyboardButton('Stacking Classifier', callback_data='g_sk'),
                   InlineKeyboardButton('Custom model', callback_data='g_custom')]

    # сборка клавиатуры из кнопок `InlineKeyboardButton`
    reply_markup = InlineKeyboardMarkup(build_menu(button_list, n_cols=3))
    # отправка клавиатуры в чат
    bot.send_message(msg.from_user.id, text=text, reply_markup=reply_markup)
    '''КНОПКА НАЗАД'''
    button_list = [InlineKeyboardButton('В меню', callback_data='back'),
                   InlineKeyboardButton('Шаг назад', callback_data='examp')]

    # сборка клавиатуры из кнопок `InlineKeyboardButton`
    reply_markup = InlineKeyboardMarkup(build_menu(button_list, n_cols=2))
    # отправка клавиатуры в чат
    bot.send_message(msg.from_user.id, text='Вернуться в меню или на шаг назад',
                     reply_markup=reply_markup)
#===========================================#
#===========================================#
def add_frac(msg, model):
        if check_num(msg.text):
            frac = int(msg.text[-2:])
            X, y = gen_data(frac)
            table, predict = models.predict_gan(model, X, y)
            tab_rep = plot_report(X, y, predict, msg.from_user.id)
            bot.send_message(msg.from_user.id, text=f'Результаты {model}:')
            bot.send_message(msg.from_user.id, text=f'{table}')
            with open(f"plot_{msg.from_user.id}.png", 'rb') as img:
                bot.send_photo(msg.from_user.id, photo=img, caption='График')
            os.remove(f"plot_{msg.from_user.id}.png")
            bot.send_message(msg.from_user.id, text=f'{tab_rep}')

            '''КНОПКА НАЗАД'''
            button_list = [InlineKeyboardButton('В меню', callback_data='back'),
                           InlineKeyboardButton('Шаг назад', callback_data='gan')]

            # сборка клавиатуры из кнопок `InlineKeyboardButton`
            reply_markup = InlineKeyboardMarkup(build_menu(button_list, n_cols=2))
            # отправка клавиатуры в чат
            bot.send_message(msg.from_user.id, text='Вернуться в меню или на шаг назад',
                             reply_markup=reply_markup)

        else:
            bot.send_message(msg.from_user.id, text='Извините, вы ввели значение в неправильном формате!\n'
                                                    'Попробуйте еще раз!')
            '''КНОПКА НАЗАД'''
            button_list = [InlineKeyboardButton('В меню', callback_data='back'),
                           InlineKeyboardButton('Шаг назад', callback_data='gan')]

            # сборка клавиатуры из кнопок `InlineKeyboardButton`
            reply_markup = InlineKeyboardMarkup(build_menu(button_list, n_cols=2))
            # отправка клавиатуры в чат
            bot.send_message(msg.from_user.id, text='Вернуться в меню или на шаг назад',
                             reply_markup=reply_markup)
#===========================================#
#===========================================#
@bot.callback_query_handler(func=lambda c: c.data == 'g_lgbm')
def button_examp(msg):
    '''Функция определения ответа на кнопку LGBM REAL'''
    msg = bot.send_message(msg.from_user.id, text='Введите долю мошеннических транзакций в данных,'
                                            ' из диапазона 0<x<1,'
                                            ' с точностью до сотых (2-x знаков после точки)\n'
                                            ' Примеры: 0.30; 0,66; 0.02')
    model = 'LGBM'
    bot.register_next_step_handler(msg, add_frac, model)
#===========================================#
#===========================================#
@bot.callback_query_handler(func=lambda c: c.data == 'g_sk')
def button_examp(msg):
    '''Функция определения ответа на кнопку LGBM REAL'''
    msg = bot.send_message(msg.from_user.id, text='Введите долю мошеннических транзакций в данных,'
                                            ' из диапазона 0<x<1,'
                                            ' с точностью до сотых (2-x знаков после точки)\n'
                                            ' Примеры: 0.30; 0,66; 0.02')
    model = 'Stacking Classifier'
    bot.register_next_step_handler(msg, add_frac, model)
#===========================================#
#===========================================#
@bot.callback_query_handler(func=lambda c: c.data == 'g_custom')
def button_examp(msg):
    '''Функция определения ответа на кнопку LGBM REAL'''
    msg = bot.send_message(msg.from_user.id, text='Введите долю мошеннических транзакций в данных,'
                                            ' из диапазона 0<x<1,'
                                            ' с точностью до сотых (2-x знаков после точки)\n'
                                            ' Примеры: 0.30; 0,66; 0.02')
    model = 'Custom'
    bot.register_next_step_handler(msg, add_frac, model)
###
###
###
###
###
###
@bot.message_handler(content_types=['text'])
def handle_docs_audio(msg):
    bot.send_message(msg.from_user.id, text="Я не понимаю :( Начните сначала")

    '''КНОПКА НАЗАД'''
    button_list = [InlineKeyboardButton('Назад', callback_data='back')]

    # сборка клавиатуры из кнопок `InlineKeyboardButton`
    reply_markup = InlineKeyboardMarkup(build_menu(button_list, n_cols=1))
    # отправка клавиатуры в чат
    bot.send_message(msg.from_user.id, text='Вернуться в главное меню',
                     reply_markup=reply_markup)
###
###
###
###
###
###
bot.polling(none_stop=True)