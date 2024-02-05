# Databricks notebook source
# MAGIC %md
# MAGIC # Python 101

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tipos de datos

# COMMAND ----------

# MAGIC %md
# MAGIC ### Cadenas de caracteres (strings)

# COMMAND ----------

s = 'a string'

# COMMAND ----------

s

# COMMAND ----------

type(s)

# COMMAND ----------

# MAGIC %md
# MAGIC Podemos formatear las cadenas de dos formas distintas: usando `f-strings` o el metodo `format()`

# COMMAND ----------

str_formatted = f'{s} is a string'

# COMMAND ----------

str_formatted

# COMMAND ----------

str_formatted_2 = '{placeholder} is a string'.format(placeholder=s)

# COMMAND ----------

str_formatted_2

# COMMAND ----------

# MAGIC %md
# MAGIC Las cadenas tienen sus propias funciones como operan sobre ellas mismas (tecnicamente son metodos del objeto):

# COMMAND ----------

str_formatted.upper() # there are many more

# COMMAND ----------

dir(str_formatted_2)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Booleanos

# COMMAND ----------

t = True

# COMMAND ----------

f = not t

# COMMAND ----------

f

# COMMAND ----------

type(f)

# COMMAND ----------

# MAGIC %md
# MAGIC Aritmetica de Booleanos:

# COMMAND ----------

t + f

# COMMAND ----------

t * f

# COMMAND ----------

t != f

# COMMAND ----------

# MAGIC %md
# MAGIC Nota: el operador de comparacion de igualdad (`==`) es distinto al de asignacion (`=`)

# COMMAND ----------

t == f

# COMMAND ----------

t and f

# COMMAND ----------

t or f

# COMMAND ----------

# MAGIC %md
# MAGIC Podemos castear booleanos a enteros:

# COMMAND ----------

int(t)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Enteros y floats

# COMMAND ----------

type(int(t))

# COMMAND ----------

i = 1

# COMMAND ----------

r = 1.3

# COMMAND ----------

type(r)

# COMMAND ----------

# MAGIC %md
# MAGIC Operaciones con numeros:

# COMMAND ----------

i / r

# COMMAND ----------

i ** r # we also have -, *, +

# COMMAND ----------

# MAGIC %md
# MAGIC ### Listas y tuplas

# COMMAND ----------

l = ['a', 1, 1.5] 

# COMMAND ----------

type(l)

# COMMAND ----------

tup = ('a', 1, 1.5)

# COMMAND ----------

type(tup)

# COMMAND ----------

# MAGIC %md
# MAGIC Slicing

# COMMAND ----------

l[0]

# COMMAND ----------

l[1:]

# COMMAND ----------

l[:-1]

# COMMAND ----------

# MAGIC %md
# MAGIC Algunas funciones/metodos

# COMMAND ----------

l.append(True)

# COMMAND ----------

l

# COMMAND ----------

l.pop()

# COMMAND ----------

# MAGIC %md
# MAGIC Mutabilidad y unpacking

# COMMAND ----------

l

# COMMAND ----------

l[1] = 2

# COMMAND ----------

l

# COMMAND ----------

len(l)

# COMMAND ----------

tup[1] = 2 # immutable 

# COMMAND ----------

letter, integer, flt = tup  # unpacking; also works for lists

# COMMAND ----------

letter

# COMMAND ----------

f'{letter}bc'[1:]  # we can also slice strings

# COMMAND ----------

empty_l = []

# COMMAND ----------

not empty_l 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Diccionarios y conjuntos

# COMMAND ----------

d = {'a': 1, 'b': 2}

# COMMAND ----------

type(d)

# COMMAND ----------

d['a']

# COMMAND ----------

d['a'] = 3

# COMMAND ----------

d

# COMMAND ----------

d.values()

# COMMAND ----------

d.keys()

# COMMAND ----------

a_set = {'a', 'b', 'b'}

# COMMAND ----------

a_set

# COMMAND ----------

type(a_set)

# COMMAND ----------

a_set = a_set.union({'c'})

# COMMAND ----------

a_set

# COMMAND ----------

# MAGIC %md
# MAGIC ## Condicionales

# COMMAND ----------

x = 5
if x >= 4:
    print('greater or equal than four')

# COMMAND ----------

if x > 5:
    print("More than 5")
elif x < 5:
    print("Less than 5")
else:
    print("Exactly 5")

# COMMAND ----------

l = []
if l: # implicit boolean
    print("not empty")
else:
    print("empty")

# COMMAND ----------

y = 4
if x >= 4 and y >= 4:
    print("both greater")

# COMMAND ----------

y = 3
if x >=4 or y >=4:
    print("at least one greater")

# COMMAND ----------

if y == 3:
    pass # do nothing

# COMMAND ----------

# MAGIC %md
# MAGIC Ternary operator

# COMMAND ----------

z = None
y = 2 if z is None else 5

# COMMAND ----------

y

# COMMAND ----------

# MAGIC %md
# MAGIC ## Captura de excepciones
# MAGIC

# COMMAND ----------

d = {'a': 1}

# COMMAND ----------

d['b']

# COMMAND ----------

print(d.get('b'))

# COMMAND ----------

try:
    print(d['b'])
except KeyError:
    print("Key 'b' not found")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bucles: for, while y list comprehensions

# COMMAND ----------

l = range(5)

# COMMAND ----------

l

# COMMAND ----------

type(l)

# COMMAND ----------

l = list(l)

# COMMAND ----------

l

# COMMAND ----------

type(l)

# COMMAND ----------

# MAGIC %md
# MAGIC For loops

# COMMAND ----------

for i in l:
    print(f'the number is: {i}')

# COMMAND ----------

# MAGIC %md
# MAGIC Break

# COMMAND ----------

for i in l:
    print(f'the number is: {i}')
    if i >= 3:
        print("breaking the loop")
        break

# COMMAND ----------

# MAGIC %md
# MAGIC Continue

# COMMAND ----------

for i in l:
    if i == 1:
        continue
    print(f"{i}")

# COMMAND ----------

for i, j in enumerate(['a', 'b']):
    print(f"index: {i} ; value {j}")

# COMMAND ----------

# MAGIC %md
# MAGIC Else en for loops

# COMMAND ----------

for i in range(5):
    if i > 5:
        break
else:
    print("Didn't break")

# COMMAND ----------

# MAGIC %md
# MAGIC While loops

# COMMAND ----------

i = 0
while i < 5:
    i += 1
    print(i)

# COMMAND ----------

i = 1
while i > 0:
    i += 1
    print(i)
    if i == 10:
        break

# COMMAND ----------

# MAGIC %md
# MAGIC List y Dict Comprehension

# COMMAND ----------

lc = [i.upper() for i in ('a', 'b')]
print(lc)

# COMMAND ----------

[i for i in ('a', 'b') if i != 'b']


# COMMAND ----------

dc = {k.lower(): v + 1 for k,v in {'A': 1, 'B': 2}.items()}
dc

# COMMAND ----------

# MAGIC %md
# MAGIC ## Input / Output (IO)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bash desde Jupyter

# COMMAND ----------

!pwd

# COMMAND ----------

!echo "Hello\nWorld" > test_file.txt

# COMMAND ----------

!ls

# COMMAND ----------

!cat test_file.txt

# COMMAND ----------

# MAGIC %md
# MAGIC Leer un archivo

# COMMAND ----------

with open('test_file.txt', 'r') as f:
    file_contents = f.read()
    print(file_contents)

# COMMAND ----------

# MAGIC %md
# MAGIC Obtener el signature y docstring

# COMMAND ----------

open?

# COMMAND ----------

# MAGIC %md
# MAGIC Escribir un archivo

# COMMAND ----------

with open('test_file.txt', 'a') as f:
    f.write('A new line')

# COMMAND ----------

!cat test_file.txt

# COMMAND ----------

# MAGIC %md
# MAGIC ## Funciones

# COMMAND ----------

# MAGIC %md
# MAGIC Argumentos posicionales

# COMMAND ----------

def f(x, y):
    return x + y

# COMMAND ----------

f(5,7)

# COMMAND ----------

pos_args = [9, 4]
f(*pos_args)

# COMMAND ----------

# MAGIC %md
# MAGIC Argumentos opcionales

# COMMAND ----------

def g(x, y, z=1, w=0):
    """Compute x + y - z + w"""
    return f(x, y) - z + w


# COMMAND ----------

g?

# COMMAND ----------

g??

# COMMAND ----------

g(5, 7)

# COMMAND ----------

g(5, 7, 2)

# COMMAND ----------

g(x=5, y=7, w=1)

# COMMAND ----------

kw_args = {'x':5 , 'y':7, 'w':1}
g(**kw_args)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Funciones lambda (anonimas)

# COMMAND ----------

f = lambda x,y: x + y

# COMMAND ----------

f(4, 5)

# COMMAND ----------

(lambda x: x+2)(3)

# COMMAND ----------

list(filter(lambda x: x < 2, [0, 1, 4 , 6]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clases y Objetos

# COMMAND ----------

class Human:
    
    """Friendly human with a name."""

    def __init__(self, name):
        self.name = name
    
    def greet(self):
        
        print(f"Hi! My name is {self.name}")

# COMMAND ----------

# instanciamos
jose = Human('jose')

# COMMAND ----------

# hacemos una llamada a un metodo
jose.greet()

# COMMAND ----------

# podemos acceder a un atributo de manera directa
jose.name

# COMMAND ----------

# podemos crear nuevos atributos de manera dinámica sin reestricciones
jose.last_name = 'perez'

# COMMAND ----------

jose.last_name

# COMMAND ----------

# utilizamos la función sum que es una función built in del lenguaje Python
sum

# COMMAND ----------

# de esta forma podemos agregarle un nuevo método al objeto en tiempo de ejecución 
jose.suma = sum

# COMMAND ----------

# este nuevo método lo podemos utilizar como cualquier otro
jose.suma([1, 2])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Importar librerias/modulos

# COMMAND ----------

import datetime

# COMMAND ----------

now = datetime.datetime.now()
now

# COMMAND ----------

from datetime import timedelta

# COMMAND ----------

now + timedelta(days=2)

# COMMAND ----------

from datetime import date as dt

# COMMAND ----------

dt(2019, 5, 13)

# COMMAND ----------

# MAGIC %md
# MAGIC Importar un modulo propio

# COMMAND ----------

from hello_world import s

# COMMAND ----------

s