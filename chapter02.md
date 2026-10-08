# 2 Programación de GPUs en CUDA

## 2.1 Introducción a CUDA C++

### 2.1.1 Compilación con NVCC

### 2.1.2 Kernels

#### 2.1.2.1 Especificación de Kernels

#### 2.1.2.2 Lanzamiento de Kernels

#### 2.1.2.3 Intrínsecos de índice de hilo y rejilla

### 2.1.3 Memoria en la computación de GPU

#### 2.1.3.1 Memoria unificada

#### 2.1.3.2 Gestión de memoria explícita

#### 2.1.3.3 Gestión de memoria y rendimiento de la aplicación

### 2.1.4 Sincronización de CPU y GPU

### 2.1.5 Todo junto

### 2.1.6 Inicialización en tiempo de ejecución

### 2.1.7 Verificación de errores en CUDA

#### 2.1.7.1 Estado de error

#### 2.1.7.2 Errores asíncronos

#### 2.1.7.3 CUDA_LOG_FILE

### 2.1.8 Funciones del dispositivo y del host

### 2.1.9 Especificadores de variables

#### 2.1.9.1 Detección de compilación para dispositivos

### 2.1.10 Agrupaciones de bloques de hilos

#### 2.1.10.1 Lanzamiento con notación de tres chevrones

## 2.2 Introducción a CUDA Python

### 2.2.1 Ecosistema de CUDA Python

#### 2.2.1.1 Uso de bibliotecas de CUDA en Python

#### 2.2.1.2 Alcance de este capítulo

#### 2.2.1.3 Configuración

#### 2.2.1.4 Ejecución de aplicaciones de CUDA Python

### 2.2.2 Kernels SIMT en Python

#### 2.2.2.1 Especificación de Kernels

#### 2.2.2.2 Lanzamiento de Kernels

#### 2.2.2.3 Intrínsecos de índice de hilo y rejilla

### 2.2.3 Memoria en la computación de GPU

#### 2.2.3.1 Instanciación de arrays en la GPU

#### 2.2.3.2 Copia de arrays entre la memoria del host y la GPU

#### 2.2.3.3 El tipo de objeto ndarray

### 2.2.4 Sincronización del CPU y la GPU

### 2.2.5 Todo junto

### 2.2.6 Verificación de errores en CUDA Python

## 2.3 Escritura de Kernels SIMT

### 2.3.1 Fundamentos de SIMT

### 2.3.2 Jerarquía de hilos

#### 2.3.2.1 Sincronización de bloques de hilos

### 2.3.3 Espacios de memoria del dispositivo de GPU

#### 2.3.3.1 Memoria global

#### 2.3.3.2 Memoria compartida

#### 2.3.3.3 Registros

#### 2.3.3.4 Memoria local

#### 2.3.3.5 Memoria constante

#### 2.3.3.6 Cachés

#### 2.3.3.7 Memoria de textura y superficie

#### 2.3.3.8 Memoria compartida distribuida

### 2.3.4 Rendimiento de la memoria

#### 2.3.4.1 Acceso a la memoria global en secuencia

#### 2.3.4.2 Patrones de acceso a la memoria compartida

### 2.3.5 Atómica

#### 2.3.5.1 Operaciones atómicas similares a `std::atomic` en C++

#### 2.3.5.2 Operaciones atómicas en memoria en Python

### 2.3.6 Grupos cooperativos

### 2.3.7 Lanzamiento y ocupación del kernel

## 2.4 Escritura de Kernels Tile

### 2.4.1 Declaraciones de kernel y función

### 2.4.2 Lanzamiento de kernels

#### 2.4.2.1 Patrón de dimensionamiento de la rejilla

### 2.4.3 Consulta de la posición del bloque

### 2.4.4 Creación de tiles

### 2.4.5 Constantes de compilación

#### 2.4.5.1 `Python` Constant[T]

#### 2.4.5.2 Literales `_ic` y `integral_constant` en C++

### 2.4.6 Carga y almacenamiento de tiles

#### 2.4.6.1 Cargas y almacenamientos de tile-space

#### 2.4.6.2 Gather y Scatter

### 2.4.7 Control de flujo

#### 2.4.7.1 Bucles

#### 2.4.7.2 Condiciones

### 2.4.8 Operaciones aritméticas y difusión

#### 2.4.8.1 Difusión

#### 2.4.8.2 Operadores aritméticos

### 2.4.9 Primitivas de tile

#### 2.4.9.1 Multiplicación de matrices

#### 2.4.9.2 Reducciones y escaneo

#### 2.4.9.3 Transposición y permutación

#### 2.4.9.4 Selección de elementos

#### 2.4.9.5 Funciones matemáticas

### 2.4.10 Operaciones atómicas de memoria

#### 2.4.10.1 Contención entre bloques

#### 2.4.10.2 Contención dentro del bloque

#### 2.4.10.3 Operaciones atómicas admitidas

### 2.4.11 Consejos de optimización

#### 2.4.11.1 Atributo `ct::hint` en C++

#### 2.4.11.2 Argumentos y palabras clave de llamada del decorador en Python

#### 2.4.11.3 Tipos de sugerencias

### 2.4.12 Consejos de rendimiento en C++

#### 2.4.12.1 Use punteros `__restrict__` para arrays en memoria

#### 2.4.12.2 Marque los punteros de array como alineados en 16 bytes

#### 2.4.12.3 Prefiera `ct::partition_view` para el acceso a la memoria

#### 2.4.12.4 Use `ct::irange` para bucles limitados

## 2.5 Ejecución asíncrona

### 2.5.1 ¿Qué es la ejecución concurrente asíncrona?

### 2.5.2 Flujos de CUDA

#### 2.5.2.1 Creación y destrucción de flujos de CUDA

#### 2.5.2.2 Lanzamiento de kernels en flujos de CUDA

#### 2.5.2.3 Lanzamiento de transferencias de memoria en flujos de CUDA

#### 2.5.2.4 Sincronización de flujos

### 2.5.3 Eventos de CUDA

#### 2.5.3.1 Creación y destrucción de eventos de CUDA

#### 2.5.3.2 Inserción de eventos en flujos de CUDA

#### 2.5.3.3 Tiempo de operaciones en flujos de CUDA

#### 2.5.3.4 Verificación del estado de los eventos de CUDA

#### 2.5.3.5 Funciones de devolución de llamada de flujos

#### 2.5.4.1 Uso de `cudaStreamAddCallback()`

#### 2.5.4.2 Manejo de errores asíncronos

### 2.5.6 Ordenamiento de flujos

### 2.5.7 Sincronización explícita

### 2.5.8 Sincronización implícita

### 2.5.9 Temas adicionales

#### 2.5.9.1 Priorización de flujos

#### 2.5.9.2 Introducción a los gráficos CUDA con captura de flujo

### 2.5.10 Resumen de la ejecución asíncrona

## 2.6 Memoria unificada y memoria del sistema

### 2.6.1 Espacio de direcciones virtual unificado

### 2.6.2 Memoria unificada

#### 2.6.2.1 Paradigmas de memoria unificada

#### 2.6.2.2 Soporte completo para la característica de memoria unificada

#### 2.6.2.3 Soporte limitado para la característica de memoria unificada

#### 2.6.2.4 Consejos y prefetch de memoria

### 2.6.3 Memoria del host bloqueada

#### 2.6.3.1 Memoria mapeada

### 2.6.4 Resumen

## 2.7 NVCC: El compilador de CUDA

### 2.7.1 Archivos de fuente y encabezados de CUDA

### 2.7.2 Flujo de trabajo de compilación de NVCC

### 2.7.3 Uso básico de NVCC

#### 2.7.3.1 Generación de PTX y Cubin por NVCC

#### 2.7.3.2 Notas sobre la compilación del código del host

#### 2.7.3.3 Compilación separada del código de GPU

### 2.7.4 Opciones de compilador comunes

#### 2.7.4.1 Características del lenguaje

#### 2.7.4.2 Opciones de depuración

#### 2.7.4.3 Opciones de optimización

#### 2.7.4.4 Optimización en tiempo de enlace (LTO)

#### 2.7.4.5 Opciones de perfilado

#### 2.7.4.6 Compresión de fatbin

#### 2.7.4.7 Controles de rendimiento del compilador
