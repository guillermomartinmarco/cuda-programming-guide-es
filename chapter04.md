# 4 Características de CUDA

## 4.1 Memoria Unificada

## 4.1.1 Memoria Unificada en Dispositivos con Soporte Completo para Memoria Unificada de CUDA

## 4.1.1.1 Memoria Unificada: Ejemplos Detallados

## 4.1.1.2 Ajuste de Rendimiento

## 4.1.2 Memoria Unificada en Dispositivos con Solo Soporte para Memoria Gestionada por CUDA

## 4.1.3 Memoria Unificada en Windows, WSL y Tegra

## 4.1.3.1 Múltiples GPUs

## 4.1.3.2 Coherencia y Concurrencia

## 4.1.3.3 Memoria Asociada a Flujo

## 4.1.4 Consejos de Rendimiento

## 4.1.4.1 Prefetching de Datos

## 4.1.4.2 Consejos de Uso de Datos

## 4.1.4.3 Descarte de Memoria

## 4.1.4.4 Consulta de Atributos de Uso de Datos en Memoria Gestionada

## 4.1.4.5 Sobrecarga de Memoria del GPU

## 4.2 Gráficos CUDA

## 4.2.1 Estructura del Gráfico

## 4.2.1.1 Tipos de Nodos

## 4.2.1.2 Datos de Borde

## 4.2.2 Construcción y Ejecución de Gráficos

## 4.2.2.1 Creación de Gráfico

## 4.2.2.2 Instanciación de Gráfico

## 4.2.2.3 Ejecución de Gráfico

## 4.2.3 Actualización de Gráficos Instanciados

## 4.2.3.1 Actualización de Gráfico Completo

## 4.2.3.2 Actualización de Nodo Individual

## 4.2.3.3 Habilitación de Nodo Individual

## 4.2.3.4 Limitaciones de Actualización de Gráfico

## 4.2.4 Nodos Condicionales

## 4.2.4.1 Manejos Condicionales

## 4.2.4.2 Requisitos del Cuerpo del Nodo Condicional

## 4.2.4.3 Nodos IF Condicionales

## 4.2.4.4 Nodos WHILE Condicionales

## 4.2.4.5 Nodos SWITCH Condicionales

## 4.2.5 Nodos de Memoria del Gráfico

## 4.2.5.1 Introducción

## 4.2.5.2 Fundamentos de la API

## 4.2.5.3 Reutilización Optimizada de la Memoria

## 4.2.5.4 Consideraciones de Rendimiento

## 4.2.5.5 Huella de Memoria Física

## 4.2.5.6 Acceso en Paralelo

## 4.2.6 Lanzamiento de Gráfico en el Dispositivo

## 4.2.6.1 Creación de Gráfico en el Dispositivo

## 4.2.6.2 Lanzamiento en el Dispositivo

## 4.2.7 Uso de APIs de Gráfico

## 4.2.8 Objetos de Usuario de CUDA

## 4.3 Allocador de Memoria Ordenado por Flujo

## 4.3.1 Introducción

## 4.3.2 Gestión de Memoria

## 4.3.2.1 Asignación de Memoria

## 4.3.2.2 Liberación de Memoria

## 4.3.3 Pools de Memoria

## 4.3.3.1 Pools Predeterminados/Implícitos

## 4.3.3.2 Pools Explícitos

## 4.3.3.3 Accesibilidad del Dispositivo para Soporte Multi-GPU

## 4.3.3.4 Habilitación de Pools de Memoria para IPC

## 4.3.4 Mejores Prácticas y Ajuste

## 4.3.4.1 Consulta de Soporte

## 4.3.4.2 Comportamiento de Caché de Página Física

## 4.3.4.3 Estadísticas de Uso de Recursos

## 4.3.4.4 Políticas de Reutilización de Memoria

## 4.3.4.5 Acciones de la API de Sincronización

## 4.3.5 Anexos

## 4.3.5.1 cudaMemcpyAsync Sensibilidad al Contexto/Dispositivo

## 4.3.5.2 Consulta cudaPointerGetAttributes

## 4.3.5.3 cudaGraphAddMemsetNode

## 4.3.5.4 Atributos de Puntero

## 4.3.5.5 Memoria Virtual de CPU

## 4.4 Grupos Cooperativos

## 4.4.1 Introducción

## 4.4.2 Manejo de Grupos Cooperativos y Funciones de Miembro

## 4.4.3 Comportamiento Predeterminado / Ejecución sin Grupo

## 4.4.3.1 Cree manejos de grupo implícitos lo antes posible

## 4.4.3.2 Solo pase los manejos de grupo por referencia

## 4.4.4 Creación de Grupos Cooperativos

## 4.4.4.1 Evite las amenazas de creación de grupo

## 4.4.5 Sincronización

## 4.4.5.1 Sync

## 4.4.5.2 Barreras

## 4.4.6 Operaciones Colectivas

## 4.4.6.1 Reduce

## 4.4.6.2 Scans

## 4.4.6.3 Invoke One

## 4.4.7 Movimiento Asíncrono de Datos

## 4.4.7.1 Requisitos de Alineación para Memcpy Async

## 4.4.8 Grupos a Gran Escala

## 4.4.8.1 Cuándo usar `cudaLaunchCooperativeKernel`

## 4.5 Lanzamiento Dependiente y Sincronización Programática

## 4.5.1 Antecedentes

## 4.5.2 Descripción de la API

## 4.5.3 Uso en Gráficos de CUDA

## 4.6 Contextos Verdes

## 4.6.1 Motivación / Cuándo Usar

## 4.6.2 Contextos Verdes: Facilidad de uso

## 4.6.3 Contextos Verdes: Recursos del Dispositivo y Descriptor de Recursos

## 4.6.4 Ejemplo de Creación de Contexto Verde

## 4.6.4.1 Paso 1: Obtener recursos GPU disponibles

## 4.6.4.2 Paso 2: Particionar recursos SM

## 4.6.4.3 Paso 2 (continuación): Agregar recursos de la cola de trabajo

## 4.6.4.4 Paso 3: Crear un Descriptor de Recursos

## 4.6.4.5 Paso 4: Crear un Contexto Verde

## 4.6.5 Contextos Verdes - Lanzamiento de trabajo

## 4.6.6 APIs Adicionales para la Ejecución

## 4.6.7 Ejemplo de Contexto Verde

## 4.7 Dominios de Localidad

## 4.7.1 Descubrimiento de Dominios de Localidad

## 4.7.2 Localización de Recursos de Computo

## 4.7.3 Asignación de Memoria del Dispositivo Localizada

## 4.7.3.1 Creación de un Pool de Memoria Localizada

## 4.7.3.2 Creación de una Asignación VMM Localizada

## 4.7.4 Consulta de Ubicación de Memoria

## 4.7.5 Guía de Programación

## 4.7.5.1 Kernels Localizados con Patrón Fork/Join

## 4.7.5.2 Gráficos de CUDA

## 4.7.5.3 Notas de Compatibilidad

## 4.8 Carga Perezosa

## 4.8.1 Introducción

## 4.8.2 Historial de Cambios

## 4.8.3 Requisitos para la Carga Perezosa

## 4.8.3.1 Requisito de Versión de la API de CUDA

## 4.8.3.2 Requisito de Versión de Controlador de CUDA

## 4.8.3.3 Requisitos del Compilador

## 4.8.3.4 Requisitos del Kernel

## 4.8.4 Uso

## 4.8.4.1 Habilitar y Deshabilitar

## 4.8.4.2 Verificar si la Carga Perezosa está habilitada en tiempo de ejecución

## 4.8.4.3 Forzar la carga de un módulo en tiempo de ejecución

## 4.8.5 Peligros Potenciales

## 4.8.5.1 Impacto en la ejecución concurrente de kernels

## 4.8.5.2 Grandes asignaciones de memoria

## 4.8.5.3 Impacto en las mediciones de rendimiento

## 4.9 Gestión de Registros de Errores

## 4.9.1 Antecedentes

## 4.9.2 Activación

## 4.9.3 Salida

## 4.9.4 Descripción de la API

## 4.9.4.1 Registro de llamadas de retorno

## 4.9.4.2 Iteradores y volcados de registros

## 4.9.4.3 Comportamiento de volcamiento

## 4.9.5 Limitaciones y problemas conocidos

## 4.10 Barreras Asíncronas

## 4.10.1 Inicialización

## 4.10.2 Fase de una Barrera: Llegada, Conteo, Finalización y Reinicio

## 4.10.2.1 Entrelazamiento de Warp

## 4.10.3 Seguimiento explícito de fases

## 4.10.4 Salida temprana

## 4.10.5 Función de finalización

## 4.10.6 Seguimiento de operaciones de memoria asíncronas

## 4.10.7 Patrón de productor-consumidor usando barreras

## 4.11 Pipelines

## 4.11.1 Inicialización

## 4.11.2 Envío de trabajo

## 4.11.3 Consumo de trabajo

## 4.11.4 Entrelazamiento de Warp

## 4.11.5 Salida temprana

## 4.11.6 Seguimiento de operaciones de memoria asíncronas

## 4.11.7 Patrón de productor-consumidor usando pipelines

## 4.12 Copias de Datos Asíncronas

## 4.12.1 Uso de LDGSTS

## 4.12.1.1 Agrupación de cargas en código condicional

## 4.12.1.2 Prefetching de datos

## 4.12.1.3 Patrón productor-consumidor a través de especialización de warp

## 4.12.2 Uso del Tensor Memory Accelerator (TMA)

## 4.12.2.1 Uso de TMA para transferir arreglos unidimensionales

## 4.12.2.2 Uso de TMA para transferir arreglos multidimensionales

## 4.12.3 Uso de STAS

## 4.13 Trabajo de Robo con Control de Lanzamiento de Cluster

## 4.13.1 Detalles de la API

## 4.13.1.1 Cancelación de bloques de hilos

## 4.13.1.2 Restricciones en la cancelación de bloques de hilos

## 4.13.2 Ejemplo: Multiplicación vector-escalar

## 4.13.2.1 Caso de uso: Bloques de hilos

## 4.13.2.2 Caso de uso: Cluster de bloques de hilos

## 4.14 Control de la Memoria L2

## 4.14.1 Reserva de la Memoria L2 para Acceso Persistente

## 4.14.2 Política de la Memoria L2 para Acceso Persistente

## 4.14.3 Propiedades de acceso a la Memoria L2

## 4.14.4 Ejemplo de persistencia de la Memoria L2

## 4.14.5 Restablecer el acceso a la Memoria L2 a normal

## 4.14.6 Gestionar la utilización de la caché L2 reservada

## 4.14.7 Consultar las propiedades de la caché L2

## 4.14.8 Controlar el tamaño de reserva de la caché L2 para el acceso persistente a la memoria

## 4.15 Dominios de Sincronización de Memoria

## 4.15.1 Interferencia de barreras de memoria

## 4.15.2 Aislar el tráfico con dominios

## 4.15.3 Uso de dominios en CUDA

## 4.16 Comunicación Interprocesos

## 4.16.1 IPC usando la API de Comunicación Interprocesos heredada

## 4.16.2 IPC usando la API de Gestión de Memoria Virtual

## 4.17 Gestión de Memoria Virtual

## 4.17.1 Preliminares

## 4.17.1.1 Definiciones

## 4.17.1.2 Consulta de soporte

## 4.17.2 Descripción de la API

## 4.17.3 Compartir de Memoria Unicasta

## 4.17.3.1 Asignación y exportación

## 4.17.3.2 Compartir e importar

## 4.17.3.3 Reservar y mapear

## 4.17.3.4 Derechos de acceso

## 4.17.3.5 Liberar la memoria

## 4.17.4 Compartir de Memoria Multicasta

## 4.17.4.1 Asignación de objetos multicasta

## 4.17.4.2 Agregar dispositivos a objetos multicasta

## 4.17.4.3 Vincular memoria a objetos multicasta

## 4.17.4.4 Usar asignaciones multicasta

## 4.17.5 Configuración avanzada

## 4.17.5.1 Tipo de memoria

## 4.17.5.2 Memoria comprimible

## 4.17.5.3 Soporte de alias virtual

## 4.17.5.4 Detalles de manejo de la API para IPC específicos del SO

## 4.18 Transporte de la Fabric de Computo

## 4.18.1 Prerrequisitos y Alcance

## 4.18.2 Preliminares

## 4.18.2.1 Definiciones

## 4.18.2.2 Consulta de soporte

## 4.18.3 Descripción de la API

## 4.18.4 Verificar la pertenencia al clúster

## 4.18.5 Creación de un punto final

## 4.18.6 Identificadores lógicos de punto final

## 4.18.6.1 Límites y alineación

## 4.18.7 Compartir un punto final

## 4.18.7.1 Confirmar la preparación

## 4.18.8 Vincular memoria

## 4.18.9 Operaciones de la Fabric

## 4.18.9.1 Operaciones de hilo y warp colectivas

## 4.18.10 Estado de finalización y error

## 4.18.10.1 Configuración de la Llegada de Barreras y el Número de Transacciones

## 4.18.11 Limpieza

## 4.19 Memoria de GPU Extendida

## 4.19.1 Preliminares

## 4.19.1.1 Plataformas EGM: Topología del sistema

## 4.19.1.2 Identificadores de Socket: ¿Qué son? ¿Cómo acceder a ellos?

## 4.19.1.3 Allocadores y soporte para EGM

## 4.19.1.4 Extensiones de gestión de memoria para las APIs actuales

## 4.19.2 Uso de la Interfaz EGM

## 4.19.2.1 Nodo único, GPU única

## 4.19.2.2 Nodo único, múltiples GPUs

## 4.19.2.3 Múltiples nodos, múltiples GPUs

## 4.20 Paralelismo Dinámico de CUDA

## 4.20.1 Introducción

## 4.20.1.1 Visión general

## 4.20.2 Entorno de ejecución

## 4.20.2.1 Grids padre e hijo

## 4.20.2.2 Alcance de las primitivas de CUDA

## 4.20.2.3 Flujos y eventos

## 4.20.2.4 Ordenamiento y concurrencia

## 4.20.3 Coherencia y consistencia de la memoria

## 4.20.3.1 Memoria global

## 4.20.3.2 Memoria mapeada

## 4.20.3.3 Memoria compartida y local

## 4.20.3.4 Memoria local

## 4.20.4 Interfaz de programación

## 4.20.4.1 Fundamentos

## 4.20.4.2 Interfaz de lenguaje C++ para CDP

## 4.20.5 Directrices de programación

## 4.20.5.1 Rendimiento

## 4.20.5.2 Restricciones y limitaciones de implementación

## 4.20.5.3 Compatibilidad e interoperabilidad

## 4.20.6 Lanzamiento del lado del dispositivo desde PTX

## 4.20.6.1 APIs de lanzamiento de kernel

## 4.20.6.2 Diseño del búfer de parámetros

## 4.21 Interoperabilidad de CUDA con APIs

## 4.21.1 Interoperabilidad gráfica

## 4.21.1.1 Interoperabilidad con OpenGL

## 4.21.1.2 Interoperabilidad con Direct3D

## 4.21.1.3 Interoperabilidad en una configuración de Interfaz de Enlace Escalable (SLI)

## 4.21.2 Interoperabilidad con recursos externos

## 4.21.2.1 Interoperabilidad con Vulkan

## 4.21.2.2 Interoperabilidad con Direct3D

## 4.21.2.3 Interoperabilidad con la Interfaz de Comunicación de Software de NVIDIA (NVSCI)

## 4.22 Acceso al punto de entrada del controlador

## 4.22.1 Introducción

## 4.22.2 Tipos de datos de función del controlador

## 4.22.3 Obtención de funciones del controlador

## 4.22.3.1 Uso de la API del controlador

## 4.22.3.2 Uso de la API del runtime

## 4.22.3.3 Obtener versiones de flujo predeterminadas por hilo

## 4.22.3.4 Acceder a nuevas características de CUDA

## 4.22.4 Directrices para cuGetProcAddress

## 4.22.4.1 Directrices para el uso de la API del runtime

## 4.22.5 Determinar las razones del fallo de cuGetProcAddress
