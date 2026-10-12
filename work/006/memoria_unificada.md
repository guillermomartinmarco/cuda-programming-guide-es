# Uso de la memoria unificada

> Capítulo **2.1.3.1. Memoria unificada**

En este ejemplo haremos uso de la función `cudaMallocManaged` para reservar zonas de memoria donde, tanto la GPU como la CPU pueden acceder de manera transparente.

Nos basaremos en el ejemplo de suma vectorial. Realizaremos la suma vectorial tanto en la GPU como en la CPU, utilizando los mismos punteros de memoria. Compararemos los resultados al final.
