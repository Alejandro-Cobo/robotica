# Práctica de Robótica y percepción computacional

## Moodle
Moodle de la asignatura: https://moodle.upm.es/titulaciones/oficiales/course/view.php?id=5810

## Estructura del proyecto:
- [Documentación de cada módulo](doc).
- Código fuente:
  - [image-segmentation](image-segmentation): test del módulo de procesamiento de imágenes y herramientas para crear imágenes de entrenamiento.
  - [lib](lib): librería con código auxiliar de procesamiento de imágenes y reconocimiento de símbolos.
  - [resources](proyectoRobotica-v3.0/resources): contiene el [dataset de entrenamiento](resources/dataset) del reconocedor de símbolos, las [imágenes de entrenamiento](resources/imgs) del segmentador de imágenes y varios [videos](resources/videos/) del recorrido del circuito.
  - [symbol-recognition](symbol-recognition): contiene una herramienta para generar la base de datos de símbolos y el test estático de los clasificadores.
## Ejecutar tests
Clonar el repositorio:
```
git clone https://github.com/Alejandro-Cobo/Robotica2019
```
Moverse al directorio Robotica2019:
```
cd Robotica2019/
```
Comando para ejecutar la simulación:
```
./runTestLineSim
```
Si no existen permisos de ejecución, utilizar el comando:
```
chmod a+x runTestLineSim
```
Ejecutar la prueba de procesamiento de imágenes:
```
python image-segmentation/prac_run.py
```
Ejecutar el test estático de reconocimiento de símbolos:
```
python symbol-recognition/static_test.py
```
