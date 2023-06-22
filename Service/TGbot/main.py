# -*- coding: utf-8 -*-
import telebot
from telebot import types
from telebot.types import InlineKeyboardButton, InlineKeyboardMarkup, CallbackQuery, ReplyKeyboardRemove
import models
from models import gen_data, gen_data_baf, plot_report
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

    try:
        # Удаляем предыдущую клавиатуру
        bot.delete_message(chat_id=msg.message.chat.id, message_id=msg.message.message_id)
    except:
        pass
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
    # Удаление кнопок клавиатуры
    #bot.edit_message_reply_markup(callback_query.message.chat.id, callback_query.message.message_id,  reply_markup=None)
    #print(callback_query.message.chat.id, callback_query.message.message_id)

    # Удаляем клавиатуру
    bot.delete_message(chat_id=callback_query.message.chat.id, message_id=callback_query.message.message_id)

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

    '''Функция определения ответа на кнопку Тестирование'''

    # Удаляем клавиатуру
    bot.delete_message(chat_id=msg.message.chat.id, message_id=msg.message.message_id)

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
def button_real(msg, text='Выберите набор данных'):
    '''Функция определения ответа на кнопку ГАН'''

    bot.delete_message(chat_id=msg.message.chat.id, message_id=msg.message.message_id)
    button_list = [InlineKeyboardButton('CCF', callback_data='R_CCF'),
                   InlineKeyboardButton('BAF', callback_data='R_BAF'),
                   InlineKeyboardButton('В меню', callback_data='back'),
                   InlineKeyboardButton('Шаг назад', callback_data='examp')]
    reply_markup = InlineKeyboardMarkup(build_menu(button_list, n_cols=2))
    # отправка клавиатуры в чат
    bot.send_message(msg.from_user.id, text=text, reply_markup=reply_markup)
#===========================================#
#===========================================#
@bot.callback_query_handler(func=lambda c: c.data == 'R_CCF')
def button_real(msg, text='Выберите модель для проверки на датасете CCF'):
    '''Функция определения ответа на кнопку Пример'''
    reply_markup = InlineKeyboardMarkup()

    button_1 = InlineKeyboardButton('LGBM', callback_data='r_lgbm_ccf')
    button_2 = InlineKeyboardButton('Custom_1', callback_data='r_custom_ccf')
    reply_markup.row(button_1, button_2)

    '''КНОПКА НАЗАД'''
    button_b1 = InlineKeyboardButton('В меню', callback_data='back')
    button_b2 = InlineKeyboardButton('Шаг назад', callback_data='real')
    reply_markup.row(button_b1, button_b2)

    # Удаляем предыдущую клавиатуру
    bot.delete_message(chat_id=msg.message.chat.id, message_id=msg.message.message_id)
    # отправка клавиатуры в чат
    bot.send_message(msg.from_user.id, text=text, reply_markup=reply_markup)
#===========================================#
#===========================================#
@bot.callback_query_handler(func=lambda c: c.data == 'R_BAF')
def button_real(msg, text='Выберите модель для проверки на датасете BAF'):
    '''Функция определения ответа на кнопку Пример'''
    reply_markup = InlineKeyboardMarkup()

    button_1 = InlineKeyboardButton('Stacking Classifier', callback_data='r_sk_baf')
    button_2 = InlineKeyboardButton('Custom_1', callback_data='r_custom_baf')
    reply_markup.row(button_1, button_2)

    '''КНОПКА НАЗАД'''
    button_b1 = InlineKeyboardButton('В меню', callback_data='back')
    button_b2 = InlineKeyboardButton('Шаг назад', callback_data='real')
    reply_markup.row(button_b1, button_b2)

    # Удаляем предыдущую клавиатуру
    bot.delete_message(chat_id=msg.message.chat.id, message_id=msg.message.message_id)
    # отправка клавиатуры в чат
    bot.send_message(msg.from_user.id, text=text, reply_markup=reply_markup)
#===========================================#
#===========================================#
@bot.callback_query_handler(func=lambda c: c.data == 'r_lgbm_ccf')
def button_r_lgbm(msg):
    '''Функция определения ответа на кнопку LGBM REAL'''
    # Удаляем предыдущую клавиатуру
    bot.delete_message(chat_id=msg.message.chat.id, message_id=msg.message.message_id)
    #print(msg)
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
                   InlineKeyboardButton('Шаг назад', callback_data='R_CCF')]

    # сборка клавиатуры из кнопок `InlineKeyboardButton`
    reply_markup = InlineKeyboardMarkup(build_menu(button_list, n_cols=2))
    # отправка клавиатуры в чат
    bot.send_message(msg.from_user.id, text='Вернуться в меню или на шаг назад',
                     reply_markup=reply_markup)
#===========================================#
#===========================================#
@bot.callback_query_handler(func=lambda c: c.data == 'r_custom_ccf')
def button_r_custom(msg):
    '''Функция определения ответа на кнопку Custom REAL'''

    model = 'Custom_1_ccf'
    tmp = models.predict_real(model)
    bot.send_message(msg.from_user.id, text='Результаты Custom_1_CCF:')
    bot.send_message(msg.from_user.id, text=f'{tmp}')
    bot.send_message(msg.from_user.id, text='Результаты получены на публичном '
                                            'датасете: https://www.kaggle.com/'
                                            'datasets/mlg-ulb/creditcardfraud?'
                                            'resource=download')
    '''КНОПКА НАЗАД'''
    button_list = [InlineKeyboardButton('В меню', callback_data='back'),
                   InlineKeyboardButton('Шаг назад', callback_data='R_CCF')]

    # сборка клавиатуры из кнопок `InlineKeyboardButton`
    reply_markup = InlineKeyboardMarkup(build_menu(button_list, n_cols=2))
    # отправка клавиатуры в чат
    bot.send_message(msg.from_user.id, text='Вернуться в меню или на шаг назад',
                     reply_markup=reply_markup)
# ===========================================#
# ===========================================#
@bot.callback_query_handler(func=lambda c: c.data == 'r_sk_baf')
def button_r_sk(msg):
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
                   InlineKeyboardButton('Шаг назад', callback_data='R_BAF')]

    # сборка клавиатуры из кнопок `InlineKeyboardButton`
    reply_markup = InlineKeyboardMarkup(build_menu(button_list, n_cols=2))
    # отправка клавиатуры в чат
    bot.send_message(msg.from_user.id, text='Вернуться в меню или на шаг назад',
                     reply_markup=reply_markup)
#===========================================#
#===========================================#
@bot.callback_query_handler(func=lambda c: c.data == 'r_custom_baf')
def button_r_custom(msg):
    '''Функция определения ответа на кнопку Custom REAL'''

    model = 'Custom_1_baf'
    tmp = models.predict_real(model)
    bot.send_message(msg.from_user.id, text='Результаты Custom_1_BAF:')
    bot.send_message(msg.from_user.id, text=f'{tmp}')
    bot.send_message(msg.from_user.id, text='Результаты получены на публичном '
                                            'датасете: https://www.kaggle.com/'
                                            'datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022?'
                                            'resource=download')
    '''КНОПКА НАЗАД'''
    button_list = [InlineKeyboardButton('В меню', callback_data='back'),
                   InlineKeyboardButton('Шаг назад', callback_data='R_BAF')]

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
def button_gan(msg, text='Выберите синтетический набор данных'):
    '''Функция определения ответа на кнопку ГАН'''
    #reply_markup = InlineKeyboardMarkup()

    #button_1 = InlineKeyboardButton('CCF', callback_data='CCF')
    #button_2 = InlineKeyboardButton('BAF', callback_data='BAF')
    #reply_markup.row(button_1, button_2)
    '''КНОПКА НАЗАД'''
    #button_b_1 = InlineKeyboardButton('В меню', callback_data='back'),
    #button_b_2 = InlineKeyboardButton('Шаг назад', callback_data='examp')
    #reply_markup.row(button_b_1, button_b_2)
    # Удаляем предыдущую клавиатуру
    bot.delete_message(chat_id=msg.message.chat.id, message_id=msg.message.message_id)
    button_list = [InlineKeyboardButton('CCF', callback_data='CCF'),
                   InlineKeyboardButton('BAF', callback_data='BAF'),
                   InlineKeyboardButton('В меню', callback_data='back'),
                   InlineKeyboardButton('Шаг назад', callback_data='examp')]
    reply_markup = InlineKeyboardMarkup(build_menu(button_list, n_cols=2))
    # отправка клавиатуры в чат
    bot.send_message(msg.from_user.id, text=text, reply_markup=reply_markup)
#===========================================#
#===========================================#
@bot.callback_query_handler(func=lambda c: c.data == 'CCF')
def button_ccf(msg, text='Выберите модель для проверки на синтетических данных CCF'):
    '''Функция определения ответа на кнопку CCF'''

    reply_markup = InlineKeyboardMarkup()

    button_1 = InlineKeyboardButton('LGBM', callback_data='g_lgbm')
    button_2 = InlineKeyboardButton('Stacking Classifier', callback_data='g_sk')
    button_3 = InlineKeyboardButton('Custom_1_CCF', callback_data='g_custom')
    reply_markup.row(button_1, button_2, button_3)

    '''КНОПКА НАЗАД'''
    button_b1 = InlineKeyboardButton('В меню', callback_data='back')
    button_b2 = InlineKeyboardButton('Шаг назад', callback_data='gan')
    reply_markup.row(button_b1, button_b2)

    # Удаляем предыдущую клавиатуру
    bot.delete_message(chat_id=msg.message.chat.id, message_id=msg.message.message_id)
    # отправка клавиатуры в чат
    bot.send_message(msg.from_user.id, text=text, reply_markup=reply_markup)

#===========================================#
#===========================================#
@bot.callback_query_handler(func=lambda c: c.data == 'BAF')
def button_baf(msg, text='Выберите модель для проверки на синтетических данных BAF'):
    '''Функция определения ответа на кнопку BAF'''

    reply_markup = InlineKeyboardMarkup()

    button_1 = InlineKeyboardButton('Stacking Classifier', callback_data='g_sk_baf')
    button_2 = InlineKeyboardButton('Custom_1_BAF', callback_data='g_custom_1_baf')
    button_3 = InlineKeyboardButton('Custom_3_BAF', callback_data='g_custom_3_baf')
    reply_markup.row(button_1, button_2, button_3)

    '''КНОПКА НАЗАД'''
    button_b1 = InlineKeyboardButton('В меню', callback_data='back')
    button_b2 = InlineKeyboardButton('Шаг назад', callback_data='gan')
    reply_markup.row(button_b1, button_b2)

    # Удаляем предыдущую клавиатуру
    bot.delete_message(chat_id=msg.message.chat.id, message_id=msg.message.message_id)
    # отправка клавиатуры в чат
    bot.send_message(msg.from_user.id, text=text, reply_markup=reply_markup)
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
                           InlineKeyboardButton('Шаг назад', callback_data='CCF')]

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
                           InlineKeyboardButton('Шаг назад', callback_data='CCF')]

            # сборка клавиатуры из кнопок `InlineKeyboardButton`
            reply_markup = InlineKeyboardMarkup(build_menu(button_list, n_cols=2))
            # отправка клавиатуры в чат
            bot.send_message(msg.from_user.id, text='Вернуться в меню или на шаг назад',
                             reply_markup=reply_markup)
#===========================================#
#===========================================#
def add_frac_baf(msg, model):
    if check_num(msg.text):
        frac = int(msg.text[-2:])
        X, y = gen_data_baf(frac)
        table, predict = models.predict_gan_baf(model, X, y)
        tab_rep = plot_report(X, y, predict, msg.from_user.id, label='baf')
        bot.send_message(msg.from_user.id, text=f'Результаты {model}:')
        bot.send_message(msg.from_user.id, text=f'{table}')
        with open(f"plot_{msg.from_user.id}.png", 'rb') as img:
            bot.send_photo(msg.from_user.id, photo=img, caption='График')
        os.remove(f"plot_{msg.from_user.id}.png")
        bot.send_message(msg.from_user.id, text=f'{tab_rep}')

        '''КНОПКА НАЗАД'''
        button_list = [InlineKeyboardButton('В меню', callback_data='back'),
                       InlineKeyboardButton('Шаг назад', callback_data='BAF')]

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
                       InlineKeyboardButton('Шаг назад', callback_data='BAF')]

        # сборка клавиатуры из кнопок `InlineKeyboardButton`
        reply_markup = InlineKeyboardMarkup(build_menu(button_list, n_cols=2))
        # отправка клавиатуры в чат
        bot.send_message(msg.from_user.id, text='Вернуться в меню или на шаг назад',
                         reply_markup=reply_markup)


# ===========================================#
# ===========================================#
@bot.callback_query_handler(func=lambda c: c.data == 'g_lgbm')
def button_g_lgbm_ccf(msg):
    '''Функция определения ответа на кнопку LGBM CCF'''
    msg = bot.send_message(msg.from_user.id, text='Введите долю мошеннических транзакций в данных,'
                                            ' из диапазона 0<x<1,'
                                            ' с точностью до сотых (2-x знаков после точки)\n'
                                            ' Примеры: 0.30; 0,66; 0.02')
    model = 'LGBM'
    bot.register_next_step_handler(msg, add_frac, model)
#===========================================#
#===========================================#
@bot.callback_query_handler(func=lambda c: c.data == 'g_sk')
def button_g_sk_ccf(msg):
    '''Функция определения ответа на кнопку Stacking Classifier CCF'''
    msg = bot.send_message(msg.from_user.id, text='Введите долю мошеннических транзакций в данных,'
                                            ' из диапазона 0<x<1,'
                                            ' с точностью до сотых (2-x знаков после точки)\n'
                                            ' Примеры: 0.30; 0,66; 0.02')
    model = 'Stacking Classifier'
    bot.register_next_step_handler(msg, add_frac, model)
#===========================================#
#===========================================#
@bot.callback_query_handler(func=lambda c: c.data == 'g_custom')
def button_g_custom_ccf(msg):
    '''Функция определения ответа на кнопку Custom CCF'''
    msg = bot.send_message(msg.from_user.id, text='Введите долю мошеннических транзакций в данных,'
                                            ' из диапазона 0<x<1,'
                                            ' с точностью до сотых (2-x знаков после точки)\n'
                                            ' Примеры: 0.30; 0,66; 0.02')
    model = 'Custom_1'
    bot.register_next_step_handler(msg, add_frac, model)
#===========================================#
#===========================================#
@bot.callback_query_handler(func=lambda c: c.data == 'g_custom_1_baf')
def button_g_custom_1_baf(msg):
    '''Функция определения ответа на кнопку Custom_1 BAF'''
    msg = bot.send_message(msg.from_user.id, text='Введите долю мошеннических транзакций в данных,'
                                            ' из диапазона 0<x<1,'
                                            ' с точностью до сотых (2-x знаков после точки)\n'
                                            ' Примеры: 0.30; 0,66; 0.02')
    model = 'Custom_1'
    bot.register_next_step_handler(msg, add_frac_baf, model)
#===========================================#
#===========================================#
@bot.callback_query_handler(func=lambda c: c.data == 'g_custom_3_baf')
def button_g_custom_3_baf(msg):
    '''Функция определения ответа на кнопку Custom_3 BAF'''
    msg = bot.send_message(msg.from_user.id, text='Введите долю мошеннических транзакций в данных,'
                                            ' из диапазона 0<x<1,'
                                            ' с точностью до сотых (2-x знаков после точки)\n'
                                            ' Примеры: 0.30; 0,66; 0.02')
    model = 'Custom_3'
    bot.register_next_step_handler(msg, add_frac_baf, model)
#===========================================#
#===========================================#

@bot.callback_query_handler(func=lambda c: c.data == 'g_sk_baf')
def button_g_sk_baf(msg):
    '''Функция определения ответа на кнопку Stacking BAF'''
    msg = bot.send_message(msg.from_user.id, text='Введите долю мошеннических транзакций в данных,'
                                            ' из диапазона 0<x<1,'
                                            ' с точностью до сотых (2-x знаков после точки)\n'
                                            ' Примеры: 0.30; 0,66; 0.02')
    model = 'Stacking Classifier'
    bot.register_next_step_handler(msg, add_frac_baf, model)
#===========================================#
#===========================================#
###
###
###
###
###
###
@bot.message_handler(content_types=['text'])
def handle_error(msg):
    #bot.send_message(msg.from_user.id, text="Я не понимаю :( Начните сначала")

    '''КНОПКА НАЗАД'''
    button_list = [InlineKeyboardButton('Назад', callback_data='back')]

    # сборка клавиатуры из кнопок `InlineKeyboardButton`
    reply_markup = InlineKeyboardMarkup(build_menu(button_list, n_cols=1))
    # отправка клавиатуры в чат
    bot.send_message(msg.from_user.id, text="Я не понимаю! Начните сначала",
                     reply_markup=reply_markup)
###
###
###
###
###
###
bot.polling(none_stop=True)