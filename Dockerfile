FROM python:3.10-slim

# Crear un directorio de trabajo
WORKDIR /code

# Copiar e instalar las dependencias primero
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copiar todo el resto del código del proyecto (incluye .env.example)
COPY . .

# 🔥 TRUCO TÉCNICO: Clonar el template para crear el .env que el código espera
RUN cp .env.example .env

# Comando para iniciar la app
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]