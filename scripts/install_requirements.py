import subprocess
import os
import importlib.metadata

def install_package(package):
    """Installs a package using pip."""
    subprocess.check_call(['pip', 'install', package])

def install_requirements():
    # Obtener la ruta al archivo requirements.txt en la raíz del proyecto
    requirements_file = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')

    # Leer el archivo requirements.txt
    with open(requirements_file) as f:
        required = f.read().splitlines()

    # Obtener la lista de paquetes instalados
    installed = {pkg.metadata['Name'] for pkg in importlib.metadata.distributions()}

    # Comparar los requisitos con los paquetes instalados
    missing = [pkg for pkg in required if pkg.lower() not in installed]

    # Instalar paquetes que faltan
    if missing:
        print("Installing missing packages...")
        for pkg in missing:
            install_package(pkg)
    else:
        print("All required packages are already installed.")
