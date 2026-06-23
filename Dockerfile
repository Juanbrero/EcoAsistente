FROM python:3.10-slim

# Crear un directorio de trabajo
WORKDIR /code

# Copiar e instalar las dependencias primero para aprovechar la cache de Docker
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copiar todo el resto del código del proyecto
COPY . .

# Comando para iniciar la app con Gunicorn en el puerto obligatorio 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]