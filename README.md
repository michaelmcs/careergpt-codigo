# CareER-GPT: código de procesamiento

Código de la tesis de maestría «CareER-GPT: Arquitectura Híbrida con LoRA sobre LLMs y Modelos Tabulares para la Generación y Rankeo de Rutas Educativas en Postulantes de la Región de Puno», Escuela de Posgrado de la Universidad Nacional del Altiplano de Puno. Autor: Michael Newton Cutipa Santi.

Un modelo tabular XGBoost estima la probabilidad de ingreso por programa y los ordena, y ese ranking calibrado entra como contexto a un modelo de lenguaje Qwen2.5-1.5B-Instruct ajustado con LoRA, que redacta la ruta educativa.

## Datos

Las bases de datos no se publican porque pertenecen a las instituciones que las brindaron. Pueden solicitarse al autor, previa autorización de dichas instituciones. Para ejecutar el código hay que colocar `ADMISION_V3.xlsx` y `SIAGIE_V3.xlsx` en la carpeta `datos/`.

## Ejecución

Con Python 3.11, instalar las dependencias con `pip install -r requirements.txt` y ejecutar desde la carpeta raíz, en este orden:

```
python pipeline/01_preprocesamiento.py      # une las bases y construye una fila por postulación
python pipeline/04_ajuste.py                # busca los hiperparámetros de XGBoost
python pipeline/03_resultados.py            # modelos, pruebas, corpus CareER-Dataset y figuras
python pipeline/05_figura_importancia.py    # figura de importancia de variables
python pipeline/06_ranking.py               # evaluación del rankeo
python pipeline/07_figuras_conceptuales.py  # figuras de LoRA, arquitectura y entrenamiento
python pipeline/08_notebook.py              # genera el cuaderno de Google Colab
python pipeline/09_complementos.py          # caracterización, tasa por promedio y calibración
python pipeline/10_diccionario.py           # diccionario de datos
```

Los resultados se guardan en `resultados/`. El ajuste con LoRA se ejecuta en Google Colab con el cuaderno `colab/CareER-GPT_LoRA_Colab.ipynb`, que pide subir el archivo `resultados/CareER_Dataset.jsonl`.
