import subprocess # librería para manejar subprocesos
import os # librería para manejar archivos

# Código de instalación de paquetes reutilizable para otros proyectos
def install_package(package):
    """Instala un paquete usando pip."""
    try:
        # Verificar si el paquete ya está instalado
        subprocess.check_call(['pip', 'show', package], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"El paquete '{package}' ya está instalado.")
    except subprocess.CalledProcessError:
        print(f"Instalando el paquete: {package}")
        subprocess.check_call(['pip', 'install', package])
        print(f"Instalación exitosa: {package}")

def install_requirements():
    """Instala los paquetes listados en requirements.txt."""
    # Obtener la ruta al archivo requirements.txt
    requirements_path = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')

    try:
        with open(requirements_path) as f:
            packages = f.read().splitlines()

        print("Iniciando la instalación de paquetes...")
        for package in packages:
            install_package(package)
        print("Todos los paquetes han sido instalados exitosamente.")
    except FileNotFoundError:
        print(f"Error: El archivo 'requirements.txt' no se encontró en {requirements_path}")
    except Exception as e:
        print(f"Error durante la instalación: {e}")

# Llamar a la función para instalar los requisitos desde el main
