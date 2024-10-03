# Vanguard A/B Test - Team 14

Este proyecto es el análisis de un test A/B realizado en el sitio web de **Vanguard**, donde se comparan dos grupos de usuarios: un **grupo de control** (interfaz tradicional) y un **grupo de test** (nueva interfaz).

### Equipo:
> [Carlos Vergara](https://www.linkedin.com/in/carlosvergaragamez/)

> [Oscar Paul Sanchez](https://www.linkedin.com/in/oscar-sanchez-riveros/)

### Enlaces:
- [Presentación en Canva](https://www.canva.com/design/DAGR9oN8BQU/XbN88uYBWu6SmMsFf0GwKQ/view?utm_content=DAGR9oN8BQU&utm_campaign=designshare&utm_medium=link&utm_source=editor)
- [Tablero en Trello](https://trello.com/b/BjGPDHFI)
- [Dashboard en Tableau](https://public.tableau.com/views/Vanguard-WebsiteRedesign/AnalisisAB?:language=es-ES&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)


---

## Información del Proyecto

### Objetivo del Proyecto
El objetivo de este proyecto es analizar los resultados de un test A/B en el sitio web de Vanguard, mediante la comparación de la interacción de usuarios con la interfaz tradicional (grupo de control) frente a una nueva interfaz (grupo de test).

### Herramientas Utilizadas:
- **Python (Pandas, Matplotlib, Seaborn):** Para el análisis y la limpieza de datos.
- **Tableau:** Para la visualización de resultados y creación de dashboards.
- **Trello:** Para la organización y distribución de tareas.

### Procesos Realizados:
1. **Extracción y Limpieza de Datos:** A partir de archivos `.txt` en `data/raw`.
2. **Análisis Estadístico:** Refutación de hipótesis basadas en el comportamiento de los usuarios.
3. **Visualización en Tableau:** Dashboards que ilustran las diferencias de rendimiento entre los grupos.

### Requisitos previos:

- **Python**: Este proyecto se hizo con la versión 3.11.9 de Python. Puedes descargarla desde la [página oficial de Python](https://www.python.org/downloads/).


---


## Instrucciones de Instalación

### Paso 1: Clonar el repositorio

Clona este repositorio en tu máquina local:
```bash
git clone https://github.com/tu_usuario/vanguard-ab-test-team-14.git
cd vanguard-ab-test-team-14
```

### Paso 2: Instalar dependencias

Al ejecutar el archivo `main.ipynb`, se instalarán automáticamente todas las dependencias necesarias. Sin embargo, también puedes instalar manualmente usando:
```bash
pip install -r requirements.txt
```

---

## Estructura del Proyecto

### Directorios Clave

- **data/raw:** Contiene los archivos de datos en formato `.txt` que serán limpiados y estructurados.
- **data/clean:** Archivos limpios generados tras la manipulación de los datos.
- **data/tableau:** Archivos finales que se usarán en los dashboards de Tableau.
- **scripts:** Contiene los archivos de código:
  - **`functions.py`:** Funciones para la limpieza y manipulación de datos.
  - **`install_requirements.py`:** Script que asegura la instalación automática de las dependencias.
- **notebooks:** Contiene el archivo principal `main.ipynb`, que realiza el procesamiento de datos y genera los archivos finales.

### Archivos Clave

- **`main.ipynb`:** El archivo principal que realiza las siguientes tareas:
  - Lee los archivos de datos en `data/raw`.
  - Limpia y transforma los datos.
  - Genera archivos en `data/clean` y `data/tableau` para su uso en Tableau.

---

## Instrucciones para Ejecutar el Proyecto

1. **Ejecutar el Jupyter Notebook:**
   Abre `main.ipynb` en Jupyter Notebook y ejecuta todas las celdas para procesar los datos.
   
   - **Nota:** Durante la ejecución, el script `install_requirements.py` comprobará si las dependencias están instaladas, y las instalará si no es así.

2. **Archivos de Salida:**
   - **Archivos Limpios:** Se guardarán en `data/clean`:
     - `df_final_demo_clean.csv`
     - `df_final_experiment_clients_clean.csv`
     - `merged_df_clean.csv`
   - **Archivos para Tableau:** Se generarán en `data/tableau`:
     - `dim_client.csv`
     - `dim_visitor.csv`
     - `dim_visit.csv`
     - `fact_process.csv`

---

### Agradecimientos:
Queremos expresar nuestro sincero agradecimiento a **Ironhack** y a todos los profesores que nos guiaron durante este proyecto. Gracias a su apoyo y recursos, hemos podido mejorar en aspectos clave como:

- **Semana 1:** Desarrollo de habilidades en **Exploratory Data Analysis (EDA)**, **limpieza de datos**, y **estadísticas inferenciales**, lo que nos permitió abordar el análisis de manera profunda y eficiente.
  
- **Semana 2:** Aplicación de herramientas de **Business Intelligence (BI)**, como **Tableau**, para la creación de dashboards interactivos y visualizaciones efectivas que apoyaron nuestras conclusiones.
  

Gracias a **Ironhack**, hemos adquirido una visión más clara y estructurada de cómo realizar un análisis de datos completo, desde la limpieza hasta la visualización, y estamos entusiasmados por aplicar estos conocimientos en futuros proyectos. 

--- 

