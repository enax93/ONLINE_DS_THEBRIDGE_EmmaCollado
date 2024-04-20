### Análisis Exploratorios de Datos: Exploración Pokémon - Descifrando Secretos a Través de los Datos

## Descripción

Este proyecto tiene como objetivo realizar un análisis exploratorio de datos sobre Pokémon utilizando Python, jupyter notebooks y varias bibliotecas como Pandas, Matplotlib y Seaborn. Se han explorado datos para descubrir patrones, tendencias y secretos ocultos dentro del vasto conjunto de datos de Pokémon, se incluye la comprensión de la distribución de tipos de Pokémon, la identificación de Pokémon legendarios o poderosos en función de sus estadísticas, el análisis de la frecuencia de aparición de ciertos tipos de Pokémon en diferentes generaciones, entre otros aspectos.

## Archivos
* Analisis.ipynb: Jupyter Notebook en donde se realizarón los diferentes análisis, graficas, comprobación de hipótesis.
* Definicion_EDA.ipynb: Jupyter Notebook en donde se define el título, tema, planteamiento de hipótesis y las primeras vistas de nuestro dataset.
* Memoria.ipynb: Jupyter Notebook que contiene el código y la narrativa del análisis exploratorio de datos.
* Limpiando_Dataset.ipynb: Jupyter Notebook en donde se realizarón las limpiezas de datos, eliminando las columnas que no utilizariamos y completando los nulos pertinentes. Se exportó lo realizado para obtener un dataset defnitivio y trabajar sobre este.
* Exploración Pokémon - Descifrando Secretos a Través de los Datos.pptx: Presentación de Microsoft PowerPoint.
* img: Carpeta en donde se encuentran las imagenes relacionadas con el proyecto.
* tools: Carpeta con defición de funciones para visualización de datos.
* data: Carpeta que contiene los datasets utilizados.
  - pokemon_actualizado.csv: Dataset actualizado, resultante de la limpieza y preprocesamiento de datos.
  - pkmn_con_nulos.csv: Extracto de los Pokémon que tienen peso y altura en nulos para ser completados.
  - pkmn_con_nulos_completados: Actualizado con los datos nulos completados.
  - pokemon.csv: Dataset original que contiene información sobre los Pokémon.

## Hipótesis
* ¿Cuales son los Pokémon más fuertes según la sumatoria de sus estadísticas? ¿Son todos legendarios?
* ¿Cómo es la distribución de los tipos de Pokémon según su genereción? ¿Cuáles son los más comunes y menos comunes tanto para su tipo principal y secundario?
* ¿Cuál es la combinación elemental más frecuente? ¿Y la menos frecuente?
* ¿Cuál es el tipo más frecuente de los Pokémon legendarios?
* ¿Cuál es el tipo más fuerte y más débil en promedio Pokémon?
* ¿Cuál es el la generación con los Pokémon mas fáciles y dificiles de capturar?
* ¿Cuáles son los Pokémon con mayor número de habilidades?
* ¿Cuáles son los Pokémon mas pesados y grandes?

## Contenido
* Limpieza de Datos: Se eliminaron los valores nulos, se corrigieron errores y se realizó una limpieza general del conjunto de datos.
* Exploración de Variables:
  - Base Total: Se identificaron los Pokémon con las estadísticas más altas.
  - Tipos de Pokémon: Se analizó la distribución de tipos de Pokémon y su relación con las estadísticas.
  - Generaciones: Se exploró la frecuencia de Pokémon por generación.
  - Habilidades: Se determinaron los Pokémon con la mayor cantidad de habilidades.
  - Tamaño y Peso: Se investigaron los Pokémon más grandes y pesados.

## Requerimientos
Python 3.x
Pandas
Matplotlib
Seaborn
Jupyter Notebook

## Ejecución
Clona este repositorio en tu máquina local.
Asegúrate de tener Python 3.x y todas las bibliotecas requeridas instaladas.
Ejecuta las celdas para reproducir el análisis y explorar los datos.
