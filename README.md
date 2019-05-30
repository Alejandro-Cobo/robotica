# Práctica de Robótica y Percepción Computacional

## Moodle
Moodle de la asignatura: https://moodle.upm.es/titulaciones/oficiales/course/view.php?id=5810

## Estructura del proyecto:
- [Documentación de cada módulo](doc).
- Código fuente:
  - [brain](brain): simulación de Pyrobot y Brain del robot.
  - [image-segmentation](image-segmentation): test del módulo de procesamiento de imágenes y herramientas para crear imágenes de entrenamiento.
  - [lib](lib): librería con código auxiliar de procesamiento de imágenes y reconocimiento de símbolos.
  - [resources](proyectoRobotica-v3.0/resources): contiene el [dataset de entrenamiento](resources/dataset) del reconocedor de símbolos, las [imágenes de entrenamiento](resources/imgs) del segmentador de imágenes y varios [videos](resources/videos/) del recorrido del circuito.
  - [symbol-recognition](symbol-recognition): contiene una herramienta para generar la base de datos de símbolos y el test estático de los clasificadores.

## Ejecutar tests
Actualizar OpenCV (la versión de la máquina virtual es antigua):
```
sudo apt install python-pip
pip install opencv-python==3.3.1.11 --user
```
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
brain/runTestLineSim
```
Si no existen permisos de ejecución, utilizar el comando:
```
chmod a+x brain/runTestLineSim
```
Ejecutar la prueba de procesamiento de imágenes:
```
python image-segmentation/prac_run.py
```
Ejecutar el test estático de reconocimiento de símbolos:
```
python symbol-recognition/static_test.py
```
## Trabajar con el robot
Conexión SSH con el ordenador del robot:
```
# Con el robot 25:
ssh -X robotica@192.168.1.25
# Con el robot 26:
ssh -X robotica@192.168.1.26
```
Crear la carpeta 'myDir' en el ordenador del robot:
```
mkdir myDir && cd myDir
```
Abrir otro terminal y transferir los archivos ejecutando el script 'update_files.sh':
```
# Con el robot 25:
./update_files.sh 25
# Con el robot 26:
./update_files.sh 26
```
Ejecutar pyrobot desde el primer terminal:
```
pyrobot -s RosServer -c rosaria.cfg -r Ros.py -b brain/BrainFollowLine.py
```
Al terminar de trabajar con el robot, ejecutar:
```
cd
rm -rf myDir
sudo shutdown -h now
```

