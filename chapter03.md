# 3 Avances en CUDA

## 3.1 APIs y Características Avanzadas de CUDA

### 3.1.1 `cudaLaunchKernelEx`

### 3.1.2 Lanzamiento de Clusters

#### 3.1.2.1 Lanzamiento con Clusters utilizando `cudaLaunchKernelEx`

#### 3.1.2.2 Bloques como Clusters

### 3.1.3 Más sobre Streams y Eventos

#### 3.1.3.1 Prioridades de Stream

#### 3.1.3.2 Sincronización Explícita

#### 3.1.3.3 Sincronización Implícita

### 3.1.4 Lanzamiento de Kernels Dependientes de Programa

### 3.1.5 Transferencias de Memoria por Lotes

### 3.1.6 Variables de Entorno

## 3.2 Programación Avanzada de Kernels

### 3.2.1 Uso de PTX

### 3.2.2 Implementación en Hardware

#### 3.2.2.1 Modelo de Ejecución SIMT

##### 3.2.2.1.1 Programación Independiente de Hilos

#### 3.2.2.2 Multithreading en Hardware

#### 3.2.2.3 Características de Ejecución Asíncrona

### 3.2.3 Escenarios de Hilo

### 3.2.4 Primitivas de Sincronización Avanzadas

#### 3.2.4.1 Atomics Escopados

#### 3.2.4.2 Barreras Asíncronas

#### 3.2.4.3 Pipelines

### 3.2.5 Copias de Datos Asíncronas

### 3.2.6 Equilibrio entre Memoria L1/Compartida

## 3.3 La API del Controlador CUDA

### 3.3.1 Contexto

### 3.3.2 Módulo

### 3.3.3 Ejecución de Kernels

### 3.3.4 Interoperabilidad entre las APIs de Runtime y Controlador

## 3.4 Programación de Sistemas con Múltiples GPUs

### 3.4.1 Contexto y Gestión de Ejecución Multi-Dispositivo

#### 3.4.1.1 Enumeración de Dispositivos

#### 3.4.1.2 Selección de Dispositivos

#### 3.4.1.3 Comportamiento de Streams, Eventos y Copias de Memoria Multi-Dispositivo

### 3.4.2 Transferencias y Acceso a Memoria Peer-to-Peer

#### 3.4.2.1 Transferencias de Memoria Peer-to-Peer

#### 3.4.2.2 Acceso a Memoria Peer-to-Peer

#### 3.4.2.3 Consistencia de Memoria Peer-to-Peer

#### 3.4.2.4 Memoria Gestionada Multi-Dispositivo

#### 3.4.2.5 Host IOMMU, Servicios de Control de Acceso PCI y VMs

#### 3.4.2.6 Transporte del Compute Fabric

## 3.5 Un Recorrido de las Características de CUDA

### 3.5.1 Mejorar el Rendimiento de los Kernels

#### 3.5.1.1 Barreras Asíncronas

#### 3.5.1.2 Copias de Datos Asíncronas y el Tensor Memory Accelerator (TMA)

#### 3.5.1.3 Pipelines

#### 3.5.1.4 Trabajo de Robo con Control de Lanzamiento de Cluster

### 3.5.2 Mejorar las Latencias

#### 3.5.2.1 Contextos Verdes

#### 3.5.2.2 Dominios de Localidad

#### 3.5.2.3 Asignación de Memoria Ordenada por Stream

#### 3.5.2.4 CUDA Graphs

#### 3.5.2.5 Lanzamiento Dependiente de Programa

#### 3.5.2.6 Carga Perezosa

### 3.5.3 Características de Funcionalidad

#### 3.5.3.1 Memoria del GPU Extendida

#### 3.5.3.2 Paralelismo Dinámico

### 3.5.4 Interoperabilidad de CUDA

#### 3.5.4.1 Interoperabilidad de CUDA con otras APIs

#### 3.5.4.2 Comunicación entre Procesos

### 3.5.5 Control Fino

#### 3.5.5.1 Gestión de Memoria Virtual

#### 3.5.5.2 Transporte del Compute Fabric

#### 3.5.5.3 Acceso a Puntos de Entrada del Controlador

#### 3.5.5.4 Gestión de Registros de Errores
