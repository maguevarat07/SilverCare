import os
import sys

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar app.py
from app import *

# Entrypoint que busca el Runtime de Python en Vercel
def handler(request, response):
    return app(request, response)
