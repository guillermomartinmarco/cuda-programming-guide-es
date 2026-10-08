# 5. Apéndices Técnicos

## 5.1 Capacidades de Computación

## 5.1.1 Obtener la capacidad de la GPU

## 5.1.2 Disponibilidad de características

## 5.1.2.1 Características específicas de la arquitectura

## 5.1.2.2 Características específicas de la familia

## 5.1.2.3 Objetivos de compilación de conjuntos de características

## 5.1.3 Características y especificaciones técnicas

## 5.2 Variables de entorno de CUDA

## 5.2.1 Enumeración y propiedades del dispositivo

## 5.2.1.1 `CUDA_VISIBLE_DEVICES`

## 5.2.1.2 `CUDA_DEVICE_ORDER`

## 5.2.1.3 `CUDA_MANAGED_FORCE_DEVICE_ALLOC`

## 5.2.2 Compilación Just-In-Time (JIT)

## 5.2.2.1 `CUDA_CACHE_DISABLE`

## 5.2.2.2 `CUDA_CACHE_PATH`

## 5.2.2.3 `CUDA_CACHE_MAXSIZE`

## 5.2.2.4 `CUDA_FORCE_PTX_JIT` y `CUDA_FORCE_JIT`

## 5.2.2.5 `CUDA_DISABLE_PTX_JIT` y `CUDA_DISABLE_JIT`

## 5.2.2.6 `CUDA_FORCE_PRELOAD_LIBRARIES`

## 5.2.3 Ejecución

## 5.2.3.1 `CUDA_LAUNCH_BLOCKING`

## 5.2.3.2 `CUDA_DEVICE_MAX_CONNECTIONS`

## 5.2.3.3 `CUDA_DEVICE_MAX_COPY_CONNECTIONS`

## 5.2.3.4 `CUDA_SCALE_LAUNCH_QUEUES`

## 5.2.3.5 `CUDA_GRAPHS_USE_NODE_PRIORITY`

## 5.2.3.6 `CUDA_DEVICE_WAITS_ON_EXCEPTION`

## 5.2.3.7 `CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT`

## 5.2.3.8 `CUDA_DISABLE_PERF_BOOST`

## 5.2.3.9 `CUDA_PREFER_SPINLOCKS_ON_HOST`

## 5.2.3.10 `CUDA_AUTO_BOOST` \[obsoleto]

## 5.2.4 Carga de Módulos

## 5.2.4.1 `CUDA_MODULE_LOADING`

## 5.2.4.2 `CUDA_MODULE_DATA_LOADING`

## 5.2.4.3 `CUDA_BINARY_LOADER_THREAD_COUNT`

## 5.2.4.4 `CUDA_DEVICE_DEFAULT_STACK_SIZE_MAX`

## 5.2.5 Gestión de Registros de Errores de CUDA

## 5.2.5.1 `CUDA_LOG_FILE`

## 5.3 Soporte de C++

## 5.3.1 Características del Lenguaje C++11

## 5.3.2 Características del Lenguaje C++14

## 5.3.3 Características del Lenguaje C++17

## 5.3.4 Características del Lenguaje C++20

## 5.3.5 Características del Lenguaje C++23

## 5.3.6 Biblioteca Estándar de C++ para CUDA

## 5.3.7 Funciones de la Biblioteca Estándar de C

## 5.3.7.1 `clock()` y `clock64()`

## 5.3.7.2 `printf()`

## 5.3.7.3 `memcpy()` y `memset()`

## 5.3.7.4 `malloc()` y `free()`

## 5.3.7.5 `alloca()`

## 5.3.8 Expresiones Lambda

## 5.3.8.1 Expresiones Lambda y Parámetros de la Función `__global__`

## 5.3.8.2 Lambda Extendidas

## 5.3.8.3 Características de Tipo Lambda Extendidas

## 5.3.8.4 Restricciones de Tipo Lambda Extendidas

## 5.3.8.5 Notas de Optimización de Lambda para Host/Dispositivo

## 5.3.8.6 Captura `*this` por Valor

## 5.3.8.7 Búsqueda Dependiente de Argumentos (ADL)

## 5.3.9 Wrappers de Funciones Polimórficas

## 5.3.10 Restricciones del Lenguaje C/C++

## 5.3.10.1 Características No Soportadas

## 5.3.10.2 Reservas de Espacios de Nombres

## 5.3.10.3 Punteros y Direcciones de Memoria

## 5.3.10.4 Variables

## 5.3.10.5 Funciones

## 5.3.10.6 Clases

## 5.3.10.7 Plantillas

## 5.3.10.8 Restricciones en el Código de Tile

## 5.3.11 Restricciones de C++11

## 5.3.11.1 `inline` Espacios de Nombres

## 5.3.11.2 `inline` Espacios de Nombres sin Nombre

## 5.3.11.3 `constexpr` Funciones

## 5.3.11.4 `constexpr` Variables

## 5.3.11.5 `__global__` Plantilla Variada

## 5.3.11.6 Funciones Predeterminadas `= default`

## 5.3.11.7 `[cuda::]std::initializer_list`

## 5.3.11.8 `[cuda::]std::move`, `[cuda::]std::forward`

## 5.3.12 Restricciones de C++14

## 5.3.12.1 Funciones con Tipo de Retorno Deducido

## 5.3.12.2 Plantillas de Variables

## 5.3.13 Restricciones de C++17

## 5.3.13.1 Variables `inline`

## 5.3.13.2 Vinculación Estructurada

## 5.3.14 Restricciones de C++20

## 5.3.14.1 Operador de Comparación de Tres Vías

## 5.3.14.2 `consteval` Funciones

## 5.3.15 Restricciones de C++23

## 5.3.15.1 Operador de Igualdad (P2468R2)

## 5.4 Extensiones del Lenguaje C/C++

## 5.4.1 Anotaciones de Funciones y Variables

## 5.4.1.1 Especificadores de Espacio de Ejecución

## 5.4.1.2 Especificadores de Espacio de Memoria

## 5.4.1.3 Especificadores de Inlining

## 5.4.1.4 Punteros `__restrict__`

## 5.4.1.5 Parámetros `__grid_constant__`

## 5.4.1.6 Resumen de Anotaciones

## 5.4.2 Tipos y Variables Incorporados

## 5.4.2.1 Extensiones de Tipo del Compilador Host

## 5.4.2.2 Variables Incorporadas

## 5.4.2.3 Tipos Incorporados

## 5.4.3 Configuración del Kernel

## 5.4.3.1 Cluster de Hilos

## 5.4.3.2 Límites de Lanzamiento

## 5.4.3.3 Número Máximo de Registros por Hilo

## 5.4.4 Primitivas de Sincronización

## 5.4.4.1 Funciones de Sincronización de Bloques de Hilos

## 5.4.4.2 Función de Sincronización de Warp

## 5.4.4.3 Funciones de Barrera de Memoria

## 5.4.5 Funciones Atómicas

## 5.4.5.1 Funciones Atómicas Legadas

## 5.4.5.2 Funciones Atómicas Incorporadas

## 5.4.6 Funciones de Warp

## 5.4.6.1 Máscara de Actividad de Warp

## 5.4.6.2 Funciones de Voto de Warp

## 5.4.6.3 Funciones de Coincidencia de Warp

## 5.4.6.4 Funciones de Reducción de Warp

## 5.4.6.5 Funciones de Intercambio de Warp

## 5.4.6.6 Restricciones para las Funciones `__sync`

## 5.4.7 Macros Específicas de CUDA

## 5.4.7.1 `__CUDA_ARCH__`

## 5.4.7.2 `__CUDA_ARCH_SPECIFIC__` y `__CUDA_ARCH_FAMILY_SPECIFIC__`

## 5.4.7.3 Macros de Pruebas de Características de CUDA

## 5.4.7.4 Atributo `__nv_pure__`

## 5.4.8 Funciones Específicas de CUDA

## 5.4.8.1 Funciones de Predicado de Espacio de Memoria

## 5.4.8.2 Funciones de Conversión de Espacio de Memoria

## 5.4.8.3 Funciones de Carga y Almacenamiento de Bajo Nivel

## 5.4.8.4 `__trap()`

## 5.4.8.5 `__nanosleep()`

## 5.4.8.6 Instrucciones de Extensión de Programación Dinámica (DPX)

## 5.4.9 Consejos de Optimización del Compilador

## 5.4.9.1 `#pragma unroll`

## 5.4.9.2 `__builtin_assume_aligned()`

## 5.4.9.3 `__builtin_assume()` y `__assume()`

## 5.4.9.4 `__builtin_constant_p()`

## 5.4.9.5 `__builtin_expect()`

## 5.4.9.6 `__builtin_unreachable()`

## 5.4.9.7 Pragmas ABI Personalizados

## 5.4.9.8 Pragma de Rendimiento de MMA

## 5.4.10 Depuración y Diagnóstico

## 5.4.10.1 Asertar

## 5.4.10.2 Función de Punto de Interrupción

## 5.4.10.3 Pragmas de Diagnóstico

## 5.4.11 Funciones de Matriz de Warp

## 5.4.11.1 Descripción

## 5.4.11.2 Punto Flotante Alternativo

## 5.4.11.3 Doble Precisión

## 5.4.11.4 Operaciones de Byte

## 5.4.11.5 Restricciones

## 5.4.11.6 Tipos de Elementos y Tamaños de Matriz

## 5.4.11.7 Ejemplo

## 5.5 Computación de Punto Flotante

## 5.5.1 Introducción al Punto Flotante

## 5.5.1.1 Formato de Punto Flotante

## 5.5.1.2 Valores Normales y Subnormales

## 5.5.1.3 Valores Especiales

## 5.5.1.4 Asociatividad

## 5.5.1.5 Multiplicación-Suma Fusión (FMA)

## 5.5.1.6 Ejemplo de Producto Punto

## 5.5.1.7 Redondeo

## 5.5.1.8 Notas sobre la Precisión de la Computación en el Host/Dispositivo

## 5.5.2 Tipos de Datos de Punto Flotante

## 5.5.3 Cumplimiento de IEEE-754 para CUDA

## 5.5.4 Cumplimiento de C/C++ para CUDA

## 5.5.5 Exposición de la Funcionalidad de Punto Flotante

## 5.5.6 Operadores Aritméticos Incorporados

## 5.5.7 Biblioteca Estándar de C++ para Matemáticas

## 5.5.7.1 Operaciones Básicas

## 5.5.7.2 Funciones Exponenciales

## 5.5.7.3 Funciones de Potencia

## 5.5.7.4 Funciones Trigonométricas

## 5.5.7.5 Funciones Hiperbólicas

## 5.5.7.6 Funciones de Error y Gamma

## 5.5.7.7 Operaciones de Punto Flotante Cercanas al Entero

## 5.5.7.8 Funciones de Manipulación de Punto Flotante

## 5.5.7.9 Funciones de Clasificación y Comparación

## 5.5.8 Funciones No Estándar de CUDA

## 5.5.9 Intrínsecas

## 5.5.9.1 Funciones Intrínsecas Básicas

## 5.5.9.2 Funciones Intrínsecas de Precisión Simple

## 5.5.9.3 Efecto de `--use_fast_math`

## 5.5.10 Referencias

## 5.6 Interfaces de Dispositivo y Intrínsecas

## 5.6.1 Primitivas de Barrera de Memoria Interfaz

## 5.6.1.1 Tipos de Datos

## 5.6.1.2 Primitivas de Barrera de Memoria API

## 5.6.2 Interfaz de Primitivas de Tubería

## 5.6.2.1 Primitiva `memcpy_async`

## 5.6.2.2 Primitiva de Compromiso

## 5.6.2.3 Primitiva de Espera

## 5.6.2.4 Primitiva de Llegada en Barrera

## 5.6.3 API de Grupos Cooperativos

## 5.6.3.1 `cooperative_groups.h`

## 5.6.3.2 `cooperative_groups/async.h`

## 5.6.3.3 `cooperative_groups/partition.h`

## 5.6.3.4 `cooperative_groups/reduce.h`

## 5.6.3.5 `cooperative_groups/scan.h`

## 5.6.3.6 `cooperative_groups/sync.h`

## 5.6.4 CUDA Device Runtime

## 5.6.4.1 Incluir la API del Device Runtime en el Código de CUDA

## 5.6.4.2 Memoria en el CUDA Device Runtime

## 5.6.4.3 ID de SM y ID de Warp

## 5.6.4.4 API de Configuración de Lanzamiento

## 5.6.4.5 Gestión del Dispositivo

## 5.6.4.6 Referencia de API

## 5.6.4.7 Errores de API y Fallos de Lanzamiento

## 5.6.4.8 Streams del Device Runtime

## 5.6.4.9 Errores ECC

## 5.7 Modelo de Memoria de C++ para CUDA

## 5.7.1 Escopos de Hilos

## 5.7.1.1 Relaciones de Escopo

## 5.7.2 Primitivas de Sincronización

## 5.7.3 Atomicidad

## 5.7.4 Carreras de Datos

## 5.7.5 Ejemplo: Mensajería

## 5.8 Modelo de Ejecución de C++ para CUDA

## 5.8.1 Hilos del Host

## 5.8.2 Hilos del Dispositivo

## 5.8.3 API de CUDA

## 5.8.3.1 Dependencias
