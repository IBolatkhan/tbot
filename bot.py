import telebot
import googlemaps
from telebot import types
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.preprocessing import Imputer

gmaps = googlemaps.Client(key='AIzaSyCGnL4NC9Cgu_oSMtLJ8GKJJtqC6JiVCd0')

rooms = types.ReplyKeyboardMarkup()
rooms.one_time_keyboard = True
rooms.row('1', '2', '3')
rooms.row('4', '5', 'Other')

regions = types.ReplyKeyboardMarkup()
regions.one_time_keyboard = True
regions.row('Zhetisuski', 'Medeuski')
regions.row('Almalinski','Auyezovski')
regions.row('Nauryzbayski', 'Turksibski')
regions.row('Bostandykski', 'Alatauski')
regions.row('Back')

floors = types.ReplyKeyboardMarkup()
floors.one_time_keyboard = True
floors.row('1','2','3', '4', '5')
floors.row('6','7','8', '9','Other')
floors.row('Back')

total_floors = types.ReplyKeyboardMarkup()
total_floors.one_time_keyboard = True
total_floors.row('4','5','6', '7', '8')
total_floors.row('9','10','11', '12','Other')
total_floors.row('Back')

typess = types.ReplyKeyboardMarkup()
typess.one_time_keyboard = True
typess.row('монолитный','панельный')
typess.row('кирпичный','каркасно-камышитовый')
typess.row('иное')

variants = types.ReplyKeyboardMarkup()
variants.one_time_keyboard = True
variants.row('Yes')
variants.row('No')

bot = telebot.TeleBot("758117330:AAF34JB0f2hctp7v9NxlLmBpaPi79C98mqo")

user_dict = {}

class User:
    def __init__(self, name):
        self.name = name
        self.region = None
        self.address = None
        self.room = None
        self.floor = None
        self.total_floor = None
        self.type = None
        self.year = None
        self.square = None
        self.miniRegion = None
        self.lat = None
        self.long = None
        self.criminology = None

@bot.message_handler(commands=['start', 'help'])
def start(message):
 bot.send_message(message.chat.id, 'Hello,'+str(message.from_user.first_name)+'! Now we are going to ask some questions to define price of your home')  
 response = bot.send_message(message.chat.id, 'Select your region', reply_markup=regions)
 bot.register_next_step_handler(response, address_func)

def address_func(message):
 if (message.text=='/exit'):
  exitBot(message)
 elif (message.text=='/start'):
  start(message)
 else:  
  chat_id = message.chat.id
  user_name = str(message.from_user.first_name) 
  user = User(user_name)
  user_dict[chat_id] = user
  user = user_dict[chat_id]
  user.region = message.text 
  response = bot.send_message(message.chat.id, 'Enter your address as --> Байзакова 252')     
  bot.register_next_step_handler(response, rooms_func)

def address_func_repeated(message):
  response = bot.send_message(message.chat.id, 'Enter your address as --> Байзакова 252')     
  bot.register_next_step_handler(response, rooms_func)

def start_func_repeated(message):
 response = bot.send_message(message.chat.id, 'Select your region', reply_markup=regions)
 bot.register_next_step_handler(response, address_func)

def rooms_func(message):
 if (message.text=='/exit'):
  exitBot(message)
 elif (message.text=='/start'):
  start(message)
 elif (message.text=='Back'):
  start_func_repeated(message)
 elif geoCode(message):
  chat_id = message.chat.id
  user = user_dict[chat_id]
  bot.send_location(message.chat.id,float(user.lat),float(user.long))
  response = bot.send_message(message.chat.id, 'Is it correct?', reply_markup=variants)
  bot.register_next_step_handler(response, checking_for_location)
 else:
  bot.send_message(message.chat.id, 'Not found, please enter again!')
  address_func_repeated(message)

def checking_for_location(message):
  if (message.text=='Yes'):
    chat_id = message.chat.id
    user = user_dict[chat_id]
    response = bot.send_message(message.chat.id, 'How many rooms do you have?', reply_markup=rooms)
    bot.register_next_step_handler(response, floor_func)
  else:
    address_func_repeated(message)

def geoCode(message):
  loc = gmaps.geocode(message.text)
  try:
    chat_id = message.chat.id
    user = user_dict[chat_id]
    user.lat = loc[0]['geometry']['location']['lat']
    user.long = loc[0]['geometry']['location']['lng']
    return True
  except Exception as e:
    return False

def rooms_func_repeated(message):
  response = bot.send_message(message.chat.id, 'How many rooms do you have?', reply_markup=rooms)
  bot.register_next_step_handler(response, floor_func)

def specify_room_func(message):
 if (message.text=='/exit'):
  exitBot(message)
 elif (message.text=='/start'):
  start(message)
 elif (message.text.isdigit()):
  chat_id = message.chat.id
  user = user_dict[chat_id]
  user.room = message.text
  response = bot.send_message(message.chat.id, 'What is the floor number?', reply_markup=floors)
  bot.register_next_step_handler(response, total_floor_func)
 else:
  response = bot.send_message(message.chat.id, 'Please, enter a number!!')
  bot.register_next_step_handler(response, specify_room_func)


def floor_func(message):
 if (message.text=='/exit'):
  exitBot(message)
 elif (message.text=='/start'):
  start(message)
 elif (message.text=='Back'):
  address_func_repeated(message)
 elif (message.text=='Other'):
  response = bot.send_message(message.chat.id, 'So, what room is this? Enter a digit.')
  bot.register_next_step_handler(response, specify_room_func)
 else:
  chat_id = message.chat.id
  user = user_dict[chat_id]
  user.room = message.text
  response = bot.send_message(message.chat.id, 'What is the floor number?', reply_markup=floors)
  bot.register_next_step_handler(response, total_floor_func)  

def floor_func_repeated(message):
  response = bot.send_message(message.chat.id, 'What is the floor number?', reply_markup=floors)
  bot.register_next_step_handler(response, total_floor_func)    

def specify_floor_func(message):
 if (message.text=='/exit'):
  exitBot(message)
 elif (message.text=='/start'):
  start(message)
 elif (message.text.isdigit()):
  chat_id = message.chat.id
  user = user_dict[chat_id]
  user.floor = message.text
  response = bot.send_message(message.chat.id, 'What is the total floor number?', reply_markup=floors)
  bot.register_next_step_handler(response, type_func)
 else:
  response = bot.send_message(message.chat.id, 'Please, enter a number!!')
  bot.register_next_step_handler(response, specify_floor_func)


def total_floor_func(message):
 if (message.text=='/exit'):
  exitBot(message)
 elif (message.text=='/start'):
  start(message)
 elif (message.text=='Back'):
  rooms_func_repeated(message)
 elif (message.text=='Other'):
  response = bot.send_message(message.chat.id, 'So, what floor is this? Enter a digit.')
  bot.register_next_step_handler(response, specify_floor_func)
 else:
  chat_id = message.chat.id
  user = user_dict[chat_id]
  user.floor = message.text
  response = bot.send_message(message.chat.id, 'What is the total floor number?', reply_markup=total_floors)
  bot.register_next_step_handler(response, type_func)  

def total_floor_func_repeated(message):
  response = bot.send_message(message.chat.id, 'What is the total floor number?', reply_markup=total_floors)
  bot.register_next_step_handler(response, type_func) 

def specify_total_floor_func(message):
 if (message.text=='/exit'):
  exitBot(message)
 elif (message.text=='/start'):
  start(message)
 elif (message.text.isdigit()):
  chat_id = message.chat.id
  user = user_dict[chat_id]
  user.total_floor = message.text
  response = bot.send_message(message.chat.id, 'What is the type of house?', reply_markup=typess)
  bot.register_next_step_handler(response, type_func)
 else:
  response = bot.send_message(message.chat.id, 'Please, enter a number!!')
  bot.register_next_step_handler(response, specify_total_floor_func)


def type_func(message):
 if (message.text=='/exit'):
  exitBot(message)
 elif (message.text=='/start'):
  start(message)
 elif (message.text=='Back'):
  floor_func_repeated(message)
 elif (message.text=='Other'):
  response = bot.send_message(message.chat.id, 'So, what total floor is this? Enter a digit.')
  bot.register_next_step_handler(response, specify_total_floor_func)
 else:
  chat_id = message.chat.id
  user = user_dict[chat_id]
  user.total_floor = message.text
  response = bot.send_message(message.chat.id, 'What is the type of house?', reply_markup=typess)
  bot.register_next_step_handler(response, year_func) 

def year_func(message):
 if (message.text=='/exit'):
  exitBot(message)
 elif (message.text=='/start'):
  start(message)
 elif (message.text=='Back'):
  total_floor_func_repeated(message)
 else:  
  chat_id = message.chat.id
  user = user_dict[chat_id]
  user.type = message.text
  response = bot.send_message(message.chat.id, 'What is the year of house?')
  bot.register_next_step_handler(response, square_func)

def square_func(message):
 if (message.text=='/exit'):
  exitBot(message)
 elif (message.text=='/start'):
  start(message)
 elif (message.text.isdigit()):
  chat_id = message.chat.id
  user = user_dict[chat_id]
  user.year = message.text
  response = bot.send_message(message.chat.id, 'What is the square of house?')
  bot.register_next_step_handler(response, end_func)
 else:
  response = bot.send_message(message.chat.id, 'Please, enter a number!!')
  bot.register_next_step_handler(response, square_func)

def end_func(message):
 if (message.text=='/exit'):
  exitBot(message)
 elif (message.text=='/start'):
  start(message)
 elif (message.text.isdigit()):
  chat_id = message.chat.id
  user = user_dict[chat_id]
  user.square = message.text
  user.miniRegion = clustering(user.lat,user.long)

  if user.region=="Medeuski":
      user.criminology = 5
  elif user.region=="Zhetisuski":
      user.criminology = 4
  elif user.region=="Auyezovski":
      user.criminology = 3
  elif user.region=="Almalinski":
      user.criminology = 3
  elif user.region=="Nauryzbayski":
      user.criminology = 5
  elif user.region=="Alatauski":
      user.criminology = 4
  elif user.region=="Turksibski":
      user.criminology = 2

  df_flat = pd.read_csv('houses_USE_IT.csv', header=None)
  df_flat.columns = ['price', 'room', 'floor',  'allFloor', 'type', 'year', 'square', 'miniRegion', 'latitude', 'longitude',  'criminalRate']

  type_mapping = {'монолитный': 1,
                  'панельный': 2,
                  'кирпичный': 3,
                  'каркасно-камышитовый':4,
                  'иное':5}
  user.type = type_mapping[user.type]

  df_flat['type'] = df_flat['type'].map(type_mapping)

  #input data
  imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
  imr = imr.fit(df_flat)
  imputed_data = imr.transform(df_flat.values)


  #divide dadatset
  X, y = imputed_data[:, 1:], imputed_data[:, 0]

  X_train, X_test, y_train, y_test = \
      train_test_split(X, y, test_size=0.2, random_state=42)
        
  #Standartize    
  stdsc = StandardScaler()
  X_train_std = stdsc.fit_transform(X_train)
  X_test_std = stdsc.transform(X_test)

  from sklearn.ensemble import RandomForestRegressor

  regr = RandomForestRegressor(max_depth=50, random_state=42)
  regr.fit(X_train, y_train)
  pred=regr.predict([[user.room,user.floor,user.total_floor,  user.type,  user.year, user.square, user.miniRegion, user.lat ,user.long, user.criminology]])
  print(pred)
  bot.send_message(message.chat.id,'predicted value is '+str(pred))
 else:
  response = bot.send_message(message.chat.id, 'Please, enter a number!!')
  bot.register_next_step_handler(response, end_func)


def clustering(lat,lng):
  lats = []
  longs = []
  lats.append(lat)
  longs.append(lng)
  with open('houses.csv', encoding='utf-8') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
          lats.append(row[8])
          longs.append(row[9])

  df = pd.DataFrame({'lat': lats, 'long': longs})

  init_array=np.array([[43.226858, 76.956171],[43.231349, 76.954605],[43.235575, 76.953615],
    [43.174855, 76.920439],[43.190596, 76.873409],[43.178801, 76.910632],
    [43.210477, 76.876190],[43.207571, 76.869493],[43.219704, 76.872655],
    [43.229382, 76.924942],[43.229172, 76.918400],[43.234042, 76.917467],
    [43.249380, 76.859087],[43.252919, 76.871201],[43.251891, 76.883369],
    [43.243029, 76.862541],[43.271386, 76.854092],[43.273262, 76.848958],
    [43.299945, 76.871853],[43.309048, 76.901363],[43.312887, 76.875584],
    [43.297271, 76.841336],[43.321317, 76.910248],[43.315959, 76.912701],
    [43.323263, 76.919663],[43.313857, 76.974235],[43.188629, 76.798768],
    [43.186729, 76.833980],[43.230154, 76.777650],[43.217295, 76.822267],
    [43.242608, 76.812190],[43.239180, 76.833987],[43.223202, 76.825100],
    [43.227383, 76.823770],[43.230022, 76.822667]])

  kmeans = KMeans(n_clusters=35, init=init_array)
  kmeans.fit(df)
  label = kmeans.predict(df)
  return label[0]

def exitBot(message):
 bot.send_message(message.chat.id, 'Bye')

@bot.message_handler(func=lambda m: True)
def echo_all(message):
 if (message.text=='/exit'):
  exitBot(message)
 elif (message.text=='/start'):
  start(message)
 elif (message.text!='/start'):
  bot.send_message(message.chat.id, 'To start using this service type \"/start\"')
bot.polling()

