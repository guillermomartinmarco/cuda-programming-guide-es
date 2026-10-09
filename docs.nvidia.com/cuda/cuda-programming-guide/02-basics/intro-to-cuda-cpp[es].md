# 2.1. Intro to CUDA C++

Este capítulo introduce algunos de los conceptos básicos del modelo de programación CUDA, ilustrando cómo se implementan en C++.

Esta guía de programación se centra en la API de tiempo de ejecución de CUDA. La API de tiempo de ejecución de CUDA es la forma más utilizada de utilizar CUDA en C++, y se basa en la API de controlador de nivel inferior de CUDA.

[API de tiempo de ejecución de CUDA y API del controlador de CUDA](../01-introduction/cuda-platform.md#cuda-platform-driver-and-runtime) analiza las diferencias entre las APIs, y [API del controlador de CUDA](../03-advanced/driver-api.md#driver-api) trata sobre cómo escribir código que combine las APIs.

Esta guía asume que el Toolkit de CUDA y los controladores de NVIDIA están instalados, y que está presente una GPU de NVIDIA compatible. Consulte [La guía rápida de CUDA](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html) para obtener instrucciones sobre la instalación de los componentes de CUDA necesarios.

## 2.1.1. Compilación con NVCC

El código de GPU escrito en C++ se compila utilizando el compilador NVIDIA Cuda, `nvcc`. `nvcc` es un motor de compilación que simplifica el proceso de compilación de código C++ o PTX: proporciona opciones sencillas y familiares de la línea de comandos y las ejecuta invocando la colección de herramientas que implementan las diferentes etapas de compilación.

Esta guía mostrará las líneas de comandos de `nvcc` que se pueden utilizar en cualquier sistema Linux con el Toolkit CUDA instalado, en una línea de comandos de Windows o PowerShell, o en el Subsistema de Windows para Linux con el Toolkit CUDA. El capítulo [sobre nvcc](nvcc.md#nvcc) de esta guía cubre los casos de uso comunes de `nvcc`, y la documentación completa se proporciona en el [manual de usuario de nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html).

## 2.1.2. Núcleos

Como se mencionó en la introducción al [Modelo de Programación CUDA](../01-introduction/programming-model.md#programming-model), las funciones que se ejecutan en la GPU y que pueden ser invocadas desde el host se denominan núcleos. Los núcleos están diseñados para ser ejecutados por múltiples hilos en paralelo simultáneamente.

### 2.1.2.1. Especificando los núcleos

El código para un núcleo se especifica utilizando el especificador `__global__`. Esto indica al compilador que esta función se compilará para la GPU de una manera que permita su invocación desde un lanzamiento de núcleo. Un lanzamiento de núcleo es una operación que inicia la ejecución de un núcleo, normalmente desde la CPU. Los núcleos son funciones con un tipo de retorno `void`.

```c
// Kernel definition
__global__ void vecAdd(float* A, float* B, float* C)
{

}
```

### 2.1.2.2. Lanzamiento de núcleos

El número de hilos que ejecutarán el núcleo en paralelo se especifica como parte del lanzamiento del núcleo. Esto se denomina configuración de ejecución. Diferentes invocaciones del mismo núcleo pueden utilizar diferentes configuraciones de ejecución, como un número diferente de hilos o bloques de hilos.

Existen dos formas de lanzar núcleos desde código de CPU, la notación de triple chevron ([notación de triple chevron](intro-to-cuda-cpp.md#intro-cpp-launching-kernels-triple-chevron)) y `cudaLaunchKernelEx`. La notación de triple chevron, que es la forma más común de lanzar núcleos, se introduce aquí. Se muestra y se analiza en detalle un ejemplo de lanzamiento de un núcleo utilizando `cudaLaunchKernelEx` en la sección [Sección 3.1.1](../03-advanced/advanced-host-programming.md#advanced-host-cudalaunchkernelex).

#### 2.1.2.2.1. Notación con tres flechas

La notación con tres flechas es una [extensión del lenguaje CUDA C++](../05-appendices/cpp-language-extensions.md#execution-configuration) que se utiliza para lanzar kernels. Se llama "notación con tres flechas" porque utiliza tres caracteres de flecha para encapsular la configuración de ejecución para el lanzamiento del kernel, es decir, `<<< >>>`. Los parámetros de configuración de ejecución se especifican como una lista separada por comas dentro de las flechas, de forma similar a los parámetros de una llamada a función. A continuación, se muestra la sintaxis para el lanzamiento del kernel `vecAdd`.

```c
 __global__ void vecAdd(float* A, float* B, float* C)
 {

 }

int main()
{
    ...
    // Kernel invocation
    vecAdd<<<1, 256>>>(A, B, C);
    ...
}
```

Los dos primeros parámetros de la notación con tres flechas son las dimensiones de la cuadrícula y las dimensiones del bloque de hilos, respectivamente. Cuando se utilizan bloques o cuadrículas de hilos unidimensionales, se pueden utilizar enteros para especificar las dimensiones.

El código anterior lanza un único bloque de hilos que contiene 256 hilos. Cada hilo ejecutará exactamente el mismo código del kernel. En [Intrínsecos de índice de hilos y cuadrículas](intro-to-cuda-cpp.md#intro-cpp-thread-indexing), mostraremos cómo cada hilo puede utilizar su índice dentro del bloque y la cuadrícula para modificar los datos en los que opera.

Existe un límite en el número de hilos por bloque, ya que todos los hilos de un bloque residen en el mismo multiprocesador de flujo (SM) y deben compartir los recursos del SM. En las GPU actuales, un bloque de hilos puede contener hasta 1024 hilos. Si los recursos lo permiten, se pueden programar varios bloques de hilos en un SM simultáneamente.

Los lanzamientos de kernels son asíncronos con respecto al hilo del host. Es decir, el kernel se configurará para la ejecución en la GPU, pero el código del host no esperará a que el kernel complete (o incluso inicie) la ejecución en la GPU antes de continuar. Se debe utilizar algún tipo de sincronización entre la GPU y la CPU para determinar que el kernel ha finalizado. La versión más básica, que sincroniza completamente toda la GPU, se muestra en [Sincronizar CPU y GPU](intro-to-cuda-cpp.md#intro-synchronizing-the-gpu). Métodos de sincronización más sofisticados se cubren en [Ejecución asíncrona](asynchronous-execution.md#asynchronous-execution).

Cuando se utilizan cuadrículas o bloques de hilos bidimensionales o tridimensionales, se utiliza el tipo de CUDA `dim3` como parámetros de las dimensiones de la cuadrícula y el bloque de hilos. El fragmento de código a continuación muestra un lanzamiento de kernel del kernel `MatAdd` utilizando una cuadrícula de bloques de hilos de 16 x 16, y cada bloque de hilos es de 8 x 8.

```c
int main()
{
    ...
    dim3 grid(16,16);
    dim3 block(8,8);
    MatAdd<<<grid, block>>>(A, B, C);
    ...
}
```

### 2.1.2.3. Intrínsecos para el índice de hilos y la cuadrícula

Dentro del código del kernel, CUDA proporciona intrínsecos para acceder a los parámetros de la configuración de ejecución y al índice de un hilo o bloque.

> - `threadIdx` proporciona el índice de un hilo dentro de su bloque de hilos. Cada hilo en un bloque de hilos tendrá un índice diferente.
> - `blockDim` proporciona las dimensiones del bloque de hilos, que se especificaron en la configuración de ejecución del lanzamiento del kernel.
> - `blockIdx` proporciona el índice de un bloque de hilos dentro de la cuadrícula. Cada bloque de hilos tendrá un índice diferente.
> - `gridDim` proporciona las dimensiones de la cuadrícula, que se especificaron en la configuración de ejecución cuando se lanzó el kernel.

Cada uno de estos intrínsecos es un vector de 3 componentes con miembros `.x`, `.y` y `.z`. Las dimensiones que no se especifican en una configuración de lanzamiento tendrán un valor predeterminado de 1.
`threadIdx` y `blockIdx` están indexados desde cero. Es decir, `threadIdx.x` tomará valores desde 0 hasta e incluyendo `blockDim.x-1`. De igual manera, `.y` y `.z` operan en sus respectivas dimensiones.

De forma similar, `blockIdx.x` tendrá valores desde 0 hasta e incluyendo `gridDim.x-1`, y lo mismo para las dimensiones `.y` y `.z`, respectivamente.

Esto permite a un hilo individual identificar el trabajo que debe realizar. Volviendo al kernel `vecAdd`, el kernel toma tres parámetros, cada uno de ellos es un vector de números de punto flotante. El kernel realiza una suma elemento por elemento de `A` y `B` y almacena el resultado en `C`. El kernel se paraleliza de manera que cada hilo realice una suma. El elemento que calcula se determina por su índice de hilo y de cuadrícula.

```c
__global__ void vecAdd(float* A, float* B, float* C)
{
   // calculate which element this thread is responsible for computing
   int workIndex = threadIdx.x + blockDim.x * blockIdx.x;

   // Perform computation
   C[workIndex] = A[workIndex] + B[workIndex];
}

int main()
{
    ...
    // A, B, and C are vectors of 1024 elements
    vecAdd<<<4, 256>>>(A, B, C);
    ...
}
```

En este ejemplo, se utilizan 4 bloques de hilos de 256 hilos para sumar un vector de 1024 elementos. En el primer bloque de hilos, `blockIdx.x` será 0, por lo que el índice de trabajo de cada hilo será simplemente su `threadIdx.x`. En el segundo bloque de hilos, `blockIdx.x` será 1, por lo que `blockDim.x * blockIdx.x` será igual a `blockDim.x`, que es 256 en este caso. El índice de trabajo para cada hilo en el segundo bloque de hilos será su `threadIdx.x + 256`. En el tercer bloque de hilos, el índice de trabajo será `threadIdx.x + 512`.

Este cálculo del `workIndex` es muy común para paralelizaciones unidimensionales. Al expandirse a dos o tres dimensiones, a menudo se sigue el mismo patrón en cada una de esas dimensiones.

#### 2.1.2.3.1. Verificación de límites

El ejemplo anterior asume que la longitud del vector es un múltiplo del tamaño del bloque de hilos, que en este caso es de 256 hilos. Para que el kernel pueda manejar cualquier longitud de vector, podemos añadir comprobaciones para asegurar que el acceso a la memoria no exceda los límites de los arrays, como se muestra a continuación, y luego lanzar un único bloque de hilos, que tendrá algunos hilos inactivos.

```c
__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
     // calculate which element this thread is responsible for computing
     int workIndex = threadIdx.x + blockDim.x * blockIdx.x;

     if(workIndex < vectorLength)
     {
         // Perform computation
         C[workIndex] = A[workIndex] + B[workIndex];
     }
}
```

Con el código del kernel anterior, se pueden lanzar más hilos de los necesarios sin provocar accesos fuera de los límites de los arrays. Cuando `workIndex` excede `vectorLength`, los hilos se terminan y no realizan ninguna tarea. Lanzar hilos adicionales en un bloque que no realiza ninguna tarea no implica un alto coste, sin embargo, se debe evitar lanzar bloques de hilos en los que no hay ningún hilo que realice tareas. Este kernel ahora puede manejar longitudes de vectores que no son múltiplos del tamaño del bloque.

El número de bloques de hilos que se necesitan se puede calcular como el entero superior del número de hilos necesarios, la longitud del vector en este caso, dividido por el número de hilos por bloque. Es decir, la división entera del número de hilos necesarios entre el número de hilos por bloque, redondeado hacia arriba. Una forma común de expresar esto como una única división entera es la siguiente. Al añadir `threads - 1` antes de la división entera, se comporta como una función de techo, añadiendo otro bloque de hilos solo si la longitud del vector no es divisible por el número de hilos por bloque.

```c
// vectorLength is an integer storing number of elements in the vector
int threads = 256;
int blocks = (vectorLength + threads-1)/threads;
vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
```

La biblioteca [CUDA Core Compute Library (CCCL)](https://nvidia.github.io/cccl/unstable/) proporciona una utilidad conveniente, `cuda::ceil_div`, para realizar esta división con redondeo, y así calcular el número de bloques necesarios para lanzar un kernel. Esta utilidad está disponible incluyendo el encabezado `<cuda/cmath>`.

```c
// vectorLength is an integer storing number of elements in the vector
int threads = 256;
int blocks = cuda::ceil_div(vectorLength, threads);
vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
```

La elección de 256 hilos por bloque aquí es arbitraria, pero a menudo es una buena opción para empezar.

## 2.1.3. Memoria en el cálculo en GPU

Para poder utilizar el kernel `vecAdd` mostrado anteriormente, los arrays `A`, `B` y `C` deben estar en memoria y accesibles para la GPU. Existen varias formas diferentes de hacerlo, y dos de ellas se ilustrarán aquí. Otros métodos se tratarán en secciones posteriores sobre [memoria unificada](understanding-memory.md#memory-unified-memory). Los espacios de memoria disponibles para el código que se ejecuta en la GPU se introdujeron en [Memoria de la GPU](../01-introduction/programming-model.md#programming-model-memory) y se detallan en [Espacios de memoria de dispositivos de la GPU](writing-cuda-kernels.md#writing-cuda-kernels-gpu-device-memory-spaces).

### 2.1.3.1. Memoria unificada

La memoria unificada es una característica del entorno de ejecución de CUDA que permite al controlador de NVIDIA gestionar el movimiento de datos entre el host y los dispositivos. La memoria se asigna utilizando la API `cudaMallocManaged` o declarando una variable con el especificador `__managed__`. El controlador de NVIDIA se asegurará de que la memoria esté accesible para la GPU o la CPU siempre que cualquiera de las dos intente acceder a ella.

El código a continuación muestra una función completa para lanzar el kernel `vecAdd`, que utiliza la memoria unificada para los vectores de entrada y salida que se utilizarán en la GPU. `cudaMallocManaged` asigna búferes que pueden ser accedidos tanto desde la CPU como desde la GPU. Estos búferes se liberan utilizando `cudaFree`.

```c
void unifiedMemExample(int vectorLength)
{
    // Pointers to memory vectors
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    // Use unified memory to allocate buffers
    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // Launch the kernel. Unified memory will make sure A, B, and C are
    // accessible to the GPU
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    // Wait for the kernel to complete execution
    cudaDeviceSynchronize();

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Unified Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");
    }

    // Clean Up
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);

}
```

La memoria unificada está soportada en todos los sistemas operativos y GPUs compatibles con CUDA, aunque el mecanismo subyacente y el rendimiento pueden variar según la arquitectura del sistema. [Memoria unificada](understanding-memory.md#memory-unified-memory) proporciona más detalles. En algunos sistemas Linux (por ejemplo, aquellos con [servicios de traducción de direcciones](understanding-memory.md#memory-unified-address-translation-services) o [gestión de memoria heterogénea](understanding-memory.md#memory-heterogeneous-memory-management)), toda la memoria del sistema se convierte automáticamente en memoria unificada, y no es necesario utilizar `cudaMallocManaged` o el especificador `__managed__`.

### 2.1.3.2. Gestión explícita de la memoria

La gestión explícita de la asignación de memoria y la migración de datos entre espacios de memoria puede ayudar a mejorar el rendimiento de la aplicación, aunque esto implica un código más extenso. El código a continuación asigna explícitamente memoria en la GPU utilizando `cudaMalloc`. La memoria en la GPU se libera utilizando la misma API `cudaFree` que se utilizó para la memoria unificada en el ejemplo anterior.

```c
void explicitMemExample(int vectorLength)
{
    // Pointers for host memory
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    // Pointers for device memory
    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;

    //Allocate Host Memory using cudaMallocHost API. This is best practice
    // when buffers will be used for copies between CPU and GPU memory
    cudaMallocHost(&A, vectorLength*sizeof(float));
    cudaMallocHost(&B, vectorLength*sizeof(float));
    cudaMallocHost(&C, vectorLength*sizeof(float));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // start-allocate-and-copy
    // Allocate memory on the GPU
    cudaMalloc(&devA, vectorLength*sizeof(float));
    cudaMalloc(&devB, vectorLength*sizeof(float));
    cudaMalloc(&devC, vectorLength*sizeof(float));

    // Copy data to the GPU
    cudaMemcpy(devA, A, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(devB, B, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemset(devC, 0, vectorLength*sizeof(float));
    // end-allocate-and-copy

    // Launch the kernel
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
    // wait for kernel execution to complete
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(C, devC, vectorLength*sizeof(float), cudaMemcpyDefault);

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Explicit Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Explicit Memory: Error - CPU and GPU answers to not match\n");
    }

    // clean up
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    free(comparisonResult);
}
```

La API de CUDA `cudaMemcpy` se utiliza para copiar datos de un búfer que reside en la CPU a un búfer que reside en la GPU. Además del puntero de destino, el puntero de origen y el tamaño en bytes, el último parámetro de `cudaMemcpy` es un `cudaMemcpyKind_t`. Esto puede tener valores como:

- `cudaMemcpyHostToDevice` para copias desde la CPU a una GPU
- `cudaMemcpyDeviceToHost` para copias desde la GPU a la CPU
- `cudaMemcpyDeviceToDevice` para copias dentro de una GPU o entre GPUs

En este ejemplo, se pasa `cudaMemcpyDefault` como el último argumento de `cudaMemcpy`. Esto hace que CUDA utilice el valor de los punteros de origen y destino para determinar el tipo de copia que se debe realizar.

La API `cudaMemcpy` es sincrónica. Es decir, no devuelve hasta que la copia se haya completado. Las copias asíncronas se introducen en [Lanzamiento de transferencias de memoria en flujos de CUDA](asynchronous-execution.md#async-execution-memory-transfers).

El código utiliza `cudaMallocHost` para asignar memoria en la CPU. Esto asigna memoria [con bloqueo de página](understanding-memory.md#memory-page-locked-host-memory) en el host, lo que puede mejorar el rendimiento de la copia y es necesario para las [transferencias de memoria asíncronas](asynchronous-execution.md#async-execution-memory-transfers). En general, es una buena práctica utilizar memoria con bloqueo de página para los búferes de la CPU que se utilizarán en las transferencias de datos a y desde las GPUs. El rendimiento puede degradarse en algunos sistemas si se bloquea demasiada memoria del host. La mejor práctica es bloquear de página solo los búferes que se utilizarán para enviar o recibir datos desde la GPU.

### 2.1.3.3. Gestión de la memoria y rendimiento de las aplicaciones

Como se puede observar en el ejemplo anterior, la gestión de la memoria explícita es más detallada, requiriendo que el programador especifique las copias entre el host y el dispositivo. Esta es la ventaja y la desventaja de la gestión de la memoria explícita: permite un mayor control sobre cuándo se copian los datos entre el host y los dispositivos, dónde reside la memoria y exactamente qué memoria se asigna en cada ubicación. La gestión de la memoria explícita puede proporcionar oportunidades de mejora del rendimiento al controlar las transferencias de memoria y superponerlas con otras computaciones.

Cuando se utiliza la memoria unificada, existen las APIs de CUDA (que se tratarán en [Consejos y pre-cargado de memoria](understanding-memory.md#memory-mem-advise-prefetch)), que proporcionan indicaciones al controlador de NVIDIA que gestiona la memoria, lo que puede permitir algunos de los beneficios de rendimiento de utilizar la gestión de la memoria explícita cuando se utiliza la memoria unificada.

## 2.1.4. Sincronización de CPU y GPU

Como se mencionó en [Lanzamiento de núcleos](intro-to-cuda-cpp.md#intro-cpp-launching-kernels), los lanzamientos de núcleos son asíncronos con respecto al hilo de la CPU que los llamó. Esto significa que el flujo de control del hilo de la CPU continuará ejecutándose antes de que el núcleo haya finalizado, e incluso posiblemente antes de que se haya iniciado. Para garantizar que un núcleo haya completado la ejecución antes de continuar en el código del host, es necesario un mecanismo de sincronización.

La forma más sencilla de sincronizar la GPU y un hilo del host es utilizando `cudaDeviceSynchronize`, que bloquea el hilo del host hasta que se haya completado todo el trabajo anterior en la GPU. En los ejemplos de este capítulo, esto es suficiente porque solo se están ejecutando operaciones individuales en la GPU. En aplicaciones más grandes, puede haber múltiples [flujos](asynchronous-execution.md#cuda-streams) ejecutando trabajo en la GPU, y `cudaDeviceSynchronize` esperará a que se complete el trabajo en todos los flujos. En estas aplicaciones, se recomienda utilizar las APIs de [Sincronización de Flujos](asynchronous-execution.md#async-execution-stream-synchronization) para sincronizar solo con un flujo específico o [Eventos CUDA](asynchronous-execution.md#cuda-events). Estos se tratarán en detalle en el capítulo de [Ejecución Asíncrona](asynchronous-execution.md#asynchronous-execution).

## 2.1.5. Uniendo todo

Las siguientes listas muestran el código completo para el kernel de suma vectorial simple introducido en este capítulo, junto con todo el código del host y las funciones de utilidad para verificar que la respuesta obtenida es correcta. Estos ejemplos utilizan de forma predeterminada una longitud de vector de 1024, pero aceptan una longitud de vector diferente como argumento de línea de comandos para el ejecutable.

**Memoria unificada**

```c
#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>

__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

void initArray(float* A, int length)
{
     std::srand(std::time({}));
    for(int i=0; i<length; i++)
    {
        A[i] = rand() / (float)RAND_MAX;
    }
}

void serialVecAdd(float* A, float* B, float* C,  int length)
{
    for(int i=0; i<length; i++)
    {
        C[i] = A[i] + B[i];
    }
}

bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001)
{
    for(int i=0; i<length; i++)
    {
        if(fabs(A[i] -B[i]) > epsilon)
        {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

//unified-memory-begin
void unifiedMemExample(int vectorLength)
{
    // Pointers to memory vectors
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    // Use unified memory to allocate buffers
    cudaMallocManaged(&A, vectorLength*sizeof(float));
    cudaMallocManaged(&B, vectorLength*sizeof(float));
    cudaMallocManaged(&C, vectorLength*sizeof(float));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // Launch the kernel. Unified memory will make sure A, B, and C are
    // accessible to the GPU
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(A, B, C, vectorLength);
    // Wait for the kernel to complete execution
    cudaDeviceSynchronize();

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Unified Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Unified Memory: Error - CPU and GPU answers do not match\n");
    }

    // Clean Up
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);

}
//unified-memory-end

int main(int argc, char** argv)
{
    int vectorLength = 1024;
    if(argc >=2)
    {
        vectorLength = std::atoi(argv[1]);
    }
    unifiedMemExample(vectorLength);
    return 0;
}
```

**Gestión de memoria explícita**

```c
#include <cuda_runtime_api.h>
#include <memory.h>
#include <cstdlib>
#include <ctime>
#include <stdio.h>
#include <cuda/cmath>

__global__ void vecAdd(float* A, float* B, float* C, int vectorLength)
{
    int workIndex = threadIdx.x + blockIdx.x*blockDim.x;
    if(workIndex < vectorLength)
    {
        C[workIndex] = A[workIndex] + B[workIndex];
    }
}

void initArray(float* A, int length)
{
     std::srand(std::time({}));
    for(int i=0; i<length; i++)
    {
        A[i] = rand() / (float)RAND_MAX;
    }
}

void serialVecAdd(float* A, float* B, float* C,  int length)
{
    for(int i=0; i<length; i++)
    {
        C[i] = A[i] + B[i];
    }
}

bool vectorApproximatelyEqual(float* A, float* B, int length, float epsilon=0.00001)
{
    for(int i=0; i<length; i++)
    {
        if(fabs(A[i] -B[i]) > epsilon)
        {
            printf("Index %d mismatch: %f != %f", i, A[i], B[i]);
            return false;
        }
    }
    return true;
}

//explicit-memory-begin
void explicitMemExample(int vectorLength)
{
    // Pointers for host memory
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    float* comparisonResult = (float*)malloc(vectorLength*sizeof(float));

    // Pointers for device memory
    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;

    //Allocate Host Memory using cudaMallocHost API. This is best practice
    // when buffers will be used for copies between CPU and GPU memory
    cudaMallocHost(&A, vectorLength*sizeof(float));
    cudaMallocHost(&B, vectorLength*sizeof(float));
    cudaMallocHost(&C, vectorLength*sizeof(float));

    // Initialize vectors on the host
    initArray(A, vectorLength);
    initArray(B, vectorLength);

    // start-allocate-and-copy
    // Allocate memory on the GPU
    cudaMalloc(&devA, vectorLength*sizeof(float));
    cudaMalloc(&devB, vectorLength*sizeof(float));
    cudaMalloc(&devC, vectorLength*sizeof(float));

    // Copy data to the GPU
    cudaMemcpy(devA, A, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(devB, B, vectorLength*sizeof(float), cudaMemcpyDefault);
    cudaMemset(devC, 0, vectorLength*sizeof(float));
    // end-allocate-and-copy

    // Launch the kernel
    int threads = 256;
    int blocks = cuda::ceil_div(vectorLength, threads);
    vecAdd<<<blocks, threads>>>(devA, devB, devC, vectorLength);
    // wait for kernel execution to complete
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(C, devC, vectorLength*sizeof(float), cudaMemcpyDefault);

    // Perform computation serially on CPU for comparison
    serialVecAdd(A, B, comparisonResult, vectorLength);

    // Confirm that CPU and GPU got the same answer
    if(vectorApproximatelyEqual(C, comparisonResult, vectorLength))
    {
        printf("Explicit Memory: CPU and GPU answers match\n");
    }
    else
    {
        printf("Explicit Memory: Error - CPU and GPU answers to not match\n");
    }

    // clean up
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    free(comparisonResult);
}
//explicit-memory-end

int main(int argc, char** argv)
{
    int vectorLength = 1024;
    if(argc >=2)
    {
        vectorLength = std::atoi(argv[1]);
    }
    explicitMemExample(vectorLength);
    return 0;
}
```

Estos se pueden construir y ejecutar utilizando nvcc de la siguiente manera:

```bash
$ nvcc vecAdd_unifiedMemory.cu -o vecAdd_unifiedMemory
$ ./vecAdd_unifiedMemory
Unified Memory: CPU and GPU answers match
$ ./vecAdd_unifiedMemory 4096
Unified Memory: CPU and GPU answers match
```

```bash
$ nvcc vecAdd_explicitMemory.cu -o vecAdd_explicitMemory
$ ./vecAdd_explicitMemory
Explicit Memory: CPU and GPU answers match
$ ./vecAdd_explicitMemory 4096
Explicit Memory: CPU and GPU answers match
```

En estos ejemplos, todos los hilos realizan tareas independientes y no necesitan coordinarse ni sincronizarse entre sí. Con frecuencia, los hilos necesitarán cooperar y comunicarse con otros hilos para llevar a cabo sus tareas. Los hilos dentro de un bloque pueden compartir datos a través de [memoria compartida](writing-cuda-kernels.md#writing-cuda-kernels-shared-memory) y sincronizarse para coordinar el acceso a la memoria.

El mecanismo más básico para la sincronización a nivel de bloque es la función intrínseca `__syncthreads()`, que actúa como una barrera en la que todos los hilos dentro del bloque deben esperar antes de que se permita que cualquier hilo continúe. [Memoria compartida](writing-cuda-kernels.md#writing-cuda-kernels-shared-memory) proporciona un ejemplo del uso de memoria compartida.

Para una cooperación eficiente, se espera que la memoria compartida sea una memoria de baja latencia cerca de cada núcleo del procesador (similar a una caché L1) y que `__syncthreads()` sea ligero. `__syncthreads()` solo sincroniza los hilos dentro de un único bloque de hilos.

La sincronización entre bloques solo está soportada en ciertas circunstancias. Por ejemplo, [grupos de bloques de hilos](../01-introduction/programming-model.md#programming-model-thread-block-clusters) permiten que los bloques dentro de un grupo se sincronicen, y las [APIs de Grupos cooperativos](../04-special-topics/cooperative-groups.md#cooperative-groups) proporcionan mecanismos para crear dominios de sincronización entre bloques.

Normalmente, el mejor rendimiento se logra cuando la sincronización se mantiene dentro de un bloque de hilos. Los bloques de hilos aún pueden trabajar en resultados comunes utilizando [funciones de memoria atómica](writing-cuda-kernels.md#writing-cuda-kernels-atomics), que se cubrirán en las secciones siguientes.

La sección [3.2.4](../03-advanced/advanced-kernel-programming.md#advanced-kernels-advanced-sync-primitives) cubre las primitivas de sincronización de CUDA que proporcionan un control muy detallado para maximizar el rendimiento y el uso de los recursos.

## 2.1.6. Inicialización en tiempo de ejecución

El entorno de ejecución de CUDA crea un [contexto de CUDA](../03-advanced/driver-api.md#driver-api-context) para cada dispositivo del sistema. Este contexto es el contexto principal para este dispositivo y se inicializa en la primera función de ejecución que requiere un contexto activo en este dispositivo. El contexto se comparte entre todos los hilos del host de la aplicación. Como parte de la creación del contexto, el código del dispositivo se [compila en tiempo real](../01-introduction/cuda-platform.md#cuda-platform-just-in-time-compilation) si es necesario y se carga en la memoria del dispositivo. Todo esto ocurre de forma transparente. El contexto principal creado por el entorno de ejecución de CUDA puede accederse a través de la API del controlador para la interoperabilidad, como se describe en [Interoperabilidad entre las API del entorno de ejecución y del controlador](../03-advanced/driver-api.md#driver-api-interop-with-runtime).

A partir de CUDA 12.0, las llamadas `cudaInitDevice` y `cudaSetDevice` inicializan el entorno de ejecución y el contexto principal [asociado](../03-advanced/driver-api.md#driver-api-context) con el dispositivo especificado. El entorno de ejecución utilizará implícitamente el dispositivo 0 y se auto-inicializará según sea necesario para procesar las solicitudes de la API del entorno de ejecución si ocurren antes de estas llamadas. Esto es importante al programar las llamadas a funciones del entorno de ejecución y al interpretar el código de error de la primera llamada al entorno de ejecución. Antes de CUDA 12.0, `cudaSetDevice` no inicializaba el entorno de ejecución.

`cudaDeviceReset` destruye el contexto principal del dispositivo actual. Si se llaman a las API del entorno de ejecución después de que se haya destruido el contexto principal, se creará un nuevo contexto principal para ese dispositivo.

> [!NOTE]
>
> Las interfaces de CUDA utilizan un estado global que se inicializa durante la iniciación del programa del host y se destruye durante la terminación del programa del host. El uso de cualquiera de estas interfaces (implícitamente o explícitamente) durante la iniciación o terminación del programa después del principal resultará en un comportamiento indefinido.
>
> A partir de CUDA 12.0, `cudaSetDevice` inicializa explícitamente el entorno de ejecución, si aún no lo ha hecho, después de cambiar el dispositivo actual para el hilo del host. En versiones anteriores de CUDA, la inicialización del entorno de ejecución en el nuevo dispositivo se retrasaba hasta que se realizaba la primera llamada al entorno de ejecución después de `cudaSetDevice`. Por lo tanto, es muy importante comprobar el valor de retorno de `cudaSetDevice` para detectar errores de inicialización.
>
> Las funciones del entorno de ejecución de las secciones de manejo de errores y gestión de versiones del manual de referencia no inicializan el entorno de ejecución.

## 2.1.7. Verificación de errores en CUDA

Cada API de CUDA devuelve un valor de un tipo enumerado, `cudaError_t`. En el código de ejemplo, estos errores a menudo no se verifican. En las aplicaciones de producción, es la mejor práctica verificar y gestionar siempre el valor de retorno de cada llamada a la API de CUDA. Cuando no hay errores, el valor devuelto es `cudaSuccess`. Muchas aplicaciones optan por implementar una macro de utilidad, como la que se muestra a continuación:

```c
#define CUDA_CHECK(expr_to_check) do {            \
    cudaError_t result  = expr_to_check;          \
    if(result != cudaSuccess)                     \
    {                                             \
        fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                         \
                __LINE__,                         \
                result,\
                cudaGetErrorString(result));      \
    }                                             \
} while(0)
```

Esta macro utiliza la API `cudaGetErrorString`, que devuelve una cadena legible por humanos que describe el significado de un valor específico de `cudaError_t`. Utilizando la macro anterior, una aplicación llamaría a las llamadas a la API de tiempo de ejecución de CUDA dentro de una macro `CUDA_CHECK(expresión)`, como se muestra a continuación:

```c
    CUDA_CHECK(cudaMalloc(&devA, vectorLength*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devB, vectorLength*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&devC, vectorLength*sizeof(float)));
```

Si alguna de estas llamadas detecta un error, este se imprimirá en `stderr` utilizando esta macro. Esta macro es común para proyectos más pequeños, pero se puede adaptar a un sistema de registro u otro mecanismo de gestión de errores en aplicaciones más grandes.

> [!NOTE]
>
> Es importante tener en cuenta que el estado de error devuelto por cualquier llamada a la API de CUDA también puede indicar un error de una operación asíncrona emitida previamente. La sección "[Manejo de errores asíncronos](asynchronous-execution.md#asynchronous-execution-error-handling)" cubre este tema en más detalle.

### 2.1.7.1. Estado de error

El entorno de ejecución de CUDA mantiene un estado `cudaError_t` para cada hilo del host. El valor predeterminado es `cudaSuccess` y se sobrescribe cada vez que se produce un error. La función `cudaGetLastError` devuelve el estado de error actual y luego lo restablece a `cudaSuccess`. Alternativamente, `cudaPeekAtLastError` devuelve el estado de error sin restablecerlo.

Las llamadas a kernel utilizando la notación de [tres chevrons](intro-to-cuda-cpp.md#intro-cpp-launching-kernels-triple-chevron) no devuelven un `cudaError_t`. Es una buena práctica verificar el estado de error inmediatamente después de las llamadas al kernel para detectar errores inmediatos en la llamada al kernel o [errores asíncronos](intro-to-cuda-cpp.md#intro-cpp-error-checking-asynchronous) antes de la llamada al kernel. Un valor de `cudaSuccess` al verificar el estado de error inmediatamente después de una llamada al kernel no significa que el kernel se haya ejecutado correctamente o incluso que haya comenzado la ejecución. Simplemente verifica que los parámetros y la configuración de ejecución pasados al entorno de ejecución no hayan provocado ningún error, y que el estado de error no sea un error anterior o asíncrono antes de que comenzara el kernel.

### 2.1.7.2. Errores asíncronos

Los kernels de CUDA y muchas APIs de tiempo de ejecución son asíncronos. Se analizarán en detalle las APIs de tiempo de ejecución de CUDA asíncronas en [Ejecución asíncrona](asynchronous-execution.md#asynchronous-execution). El estado de error de CUDA se establece y se sobrescribe cada vez que se produce un error. Esto significa que los errores que ocurren durante la ejecución de operaciones asíncronas solo se informarán cuando se examine el estado de error. Como se ha mencionado, esto podría ser una llamada a `cudaGetLastError`, `cudaPeekAtLastError`, o podría ser cualquier API de CUDA que devuelva `cudaError_t`.

Cuando las funciones de las APIs de tiempo de ejecución de CUDA devuelven errores, el estado de error no se borra. Esto significa que el código de error de un error asíncrono, como un acceso de memoria inválido por parte de un kernel, se devolverá por cada API de tiempo de ejecución de CUDA hasta que se haya borrado el estado de error llamando a `cudaGetLastError`.

```c
    vecAdd<<<blocks, threads>>>(devA, devB, devC);
    // check error state after kernel launch
    CUDA_CHECK(cudaGetLastError());
    // wait for kernel execution to complete
    // The CUDA_CHECK will report errors that occurred during execution of the kernel
    CUDA_CHECK(cudaDeviceSynchronize());
```

> [!NOTE]
>
> El valor `cudaError_t` `cudaErrorNotReady`, que puede ser devuelto por `cudaStreamQuery` y `cudaEventQuery`, no se considera un error y no se informa mediante `cudaPeekAtLastError` o `cudaGetLastError`.

### 2.1.7.3. `CUDA_LOG_FILE`

Otra forma eficaz de identificar errores de CUDA es utilizando la variable de entorno `CUDA_LOG_FILE`. Cuando se establece esta variable de entorno, el controlador de CUDA escribirá los mensajes de error detectados en un archivo cuyo nombre de ruta se especifica en la variable de entorno. Por ejemplo, considere el siguiente código de CUDA incorrecto, que intenta iniciar un bloque de hilos que es mayor que el máximo soportado por cualquier arquitectura.

```c
__global__ void k()
{ }

int main()
{
        k<<<8192, 4096>>>(); // Invalid block size
        CUDA_CHECK(cudaGetLastError());
        return 0;
}
```

Building and running this, the check after the kernel launch detects and reports the error using the macros illustrated in [Section 2.1.7](intro-to-cuda-cpp.md#intro-cpp-error-checking).

```bash
$ nvcc errorLogIllustration.cu -o errlog
$ ./errlog
CUDA Runtime Error: /home/cuda/intro-cpp/errorLogIllustration.cu:24:1 = invalid argument
```

Sin embargo, cuando la aplicación se ejecuta con `CUDA_LOG_FILE` configurado para un archivo de texto, ese archivo contiene información adicional sobre el error.

```bash
$ env CUDA_LOG_FILE=cudaLog.txt ./errlog
CUDA Runtime Error: /home/cuda/intro-cpp/errorLogIllustration.cu:24:1 = invalid argument
$ cat cudaLog.txt
[12:46:23.854][137216133754880][CUDA][E] One or more of block dimensions of (4096,1,1) exceeds corresponding maximum value of (1024,1024,64)
[12:46:23.854][137216133754880][CUDA][E] Returning 1 (CUDA_ERROR_INVALID_VALUE) from cuLaunchKernel
```

Setting `CUDA_LOG_FILE` to `stdout` or `stderr` will print to standard out and standard error, respectively. Using the `CUDA_LOG_FILE` environment variable, it is possible to capture and identify CUDA errors, even if the application does not implement proper error checking on CUDA return values. This approach can be extremely powerful for debugging, but the environment variable alone does not allow an application to handle and recover from CUDA errors at runtime. The [error log management](../04-special-topics/error-log-management.md#error-log-management) feature of CUDA also allows a callback function to be registered with the driver which will be called whenever an error is detected. This can be used to capture and handle errors at runtime, and also to integrate CUDA error logging seamlessly into an application’s existing logging system.

[Sección 4.9](../04-special-topics/error-log-management.md#error-log-management) muestra más ejemplos de la función de gestión de registros de errores de CUDA. La gestión de registros de errores y `CUDA_LOG_FILE` están disponibles con la versión del controlador NVIDIA r570 y posteriores.

## 2.1.8. Funciones del dispositivo y del anfitrión

El especificador `__global__` se utiliza para indicar el punto de entrada para un kernel. Es decir, una función que se invocará para la ejecución en paralelo en la GPU. En la mayoría de los casos, los kernels se inician desde el anfitrión, pero es posible iniciar un kernel desde dentro de otro kernel utilizando la [paralelización dinámica](../04-special-topics/dynamic-parallelism.md#cuda-dynamic-parallelism).

El especificador `__device__` indica que una función debe compilarse para la GPU y poder ser llamada desde otras funciones `__device__` o `__global__`. Una función, incluyendo funciones de miembros de clase, funciones functor y lambdas, puede especificarse tanto como `__device__` como `__host__`, como se muestra en el siguiente ejemplo.

## 2.1.9. Especificadores de variables

Los especificadores de [CUDA](../05-appendices/cpp-language-extensions.md#memory-space-specifiers) pueden utilizarse en las declaraciones de variables estáticas para controlar su ubicación.

- `__device__` especifica que una variable se almacena en la [memoria global](writing-cuda-kernels.md#writing-cuda-kernels-global-memory)
- `__constant__` especifica que una variable se almacena en la [memoria constante](writing-cuda-kernels.md#writing-cuda-kernels-constant-memory)
- `__managed__` especifica que una variable se almacena como [memoria unificada](understanding-memory.md#memory-unified-memory)
- `__shared__` especifica que una variable se almacena en la [memoria compartida](writing-cuda-kernels.md#writing-cuda-kernels-shared-memory)

Cuando una variable se declara sin ningún especificador dentro de una función `__device__` o `__global__`, se asigna a los registros siempre que sea posible, y a la [memoria local](writing-cuda-kernels.md#writing-cuda-kernels-local-memory) cuando sea necesario. Cualquier variable que se declare sin ningún especificador fuera de una función `__device__` o `__global__` se asignará en la memoria del sistema.

### 2.1.9.1. Detección de la compilación del dispositivo

Cuando una función se especifica con `__host__ __device__`, se le indica al compilador que genere tanto código para la GPU como para la CPU para esta función. En estos casos, puede ser deseable utilizar el preprocesador para especificar el código únicamente para la versión de GPU o de CPU de la función. Verificar si `__CUDA_ARCH__` está definido es la forma más común de hacerlo, como se muestra en el ejemplo a continuación.

## 2.1.10. Agrupaciones de bloques de hilos

A partir de la capacidad de cálculo 9.0, el modelo de programación CUDA incluye un nivel jerárquico opcional llamado agrupaciones de bloques de hilos, que están formadas por bloques de hilos. De manera similar a cómo los hilos dentro de un bloque de hilos están garantizados para ser programados simultáneamente en un multiprocesador de flujo, los bloques de hilos dentro de una agrupación también están garantizados para ser programados simultáneamente en un Grupo de Procesamiento de GPU (GPC) en la GPU.

De manera similar a los bloques de hilos, las agrupaciones también se organizan en una rejilla de una, dos o tres dimensiones de agrupaciones de bloques de hilos, como se muestra en la [Figura 5](../01-introduction/programming-model.md#f005).

El número de bloques de hilos en una agrupación puede ser definido por el usuario, y se admite un máximo de 8 bloques de hilos en una agrupación como un tamaño de agrupación portátil en CUDA.
Tenga en cuenta que, en el hardware de GPU o en configuraciones MIG que son demasiado pequeñas para admitir 8 multiprocesadores, el tamaño máximo de la agrupación se reducirá en consecuencia. La identificación de estas configuraciones más pequeñas, así como de configuraciones más grandes que admitan un tamaño de agrupación de bloques de hilos superior a 8, es específica de la arquitectura y se puede consultar utilizando la API `cudaOccupancyMaxPotentialClusterSize`.

Todos los bloques de hilos dentro de la agrupación están garantizados para ser programados simultáneamente en un único Grupo de Procesamiento de GPU (GPC) y permiten que los bloques de hilos dentro de la agrupación realicen la sincronización soportada por el hardware utilizando la API `cluster.sync()` de [grupos cooperativos](../04-special-topics/cooperative-groups.md#cooperative-groups). La función de grupo también proporciona funciones de miembro para consultar el tamaño del grupo en términos de número de hilos o número de bloques utilizando las API `num_threads()` y `num_blocks()` respectivamente. El rango de un hilo o bloque dentro del grupo se puede consultar utilizando las API `dim_threads()` y `dim_blocks()` respectivamente.

Los bloques de hilos que pertenecen a una agrupación tienen acceso a la memoria compartida distribuida, que es la memoria compartida combinada de todos los bloques de hilos dentro de la agrupación. Los bloques de hilos dentro de una agrupación tienen la capacidad de leer, escribir y realizar operaciones atómicas en cualquier dirección dentro de la memoria compartida distribuida. [Memoria compartida distribuida](writing-cuda-kernels.md#writing-cuda-kernels-distributed-shared-memory) proporciona un ejemplo de cómo realizar histogramas en memoria compartida distribuida.

> [!NOTE]
>
> En un kernel lanzado utilizando el soporte de agrupación, la variable `gridDim` todavía denota el tamaño en términos de número de bloques de hilos, con fines de compatibilidad. El rango de un bloque en una agrupación se puede encontrar utilizando la API [Grupos Cooperativos](../04-special-topics/cooperative-groups.md#cooperative-groups).

### 2.1.10.1. Lanzamiento con grupos utilizando la notación de tres flechas

Un grupo de hilos puede habilitarse en un kernel utilizando un atributo del kernel en tiempo de compilación, mediante `__cluster_dims__(X,Y,Z)`, o utilizando la API de lanzamiento de kernel de CUDA `cudaLaunchKernelEx`. El ejemplo a continuación muestra cómo lanzar un grupo utilizando un atributo del kernel en tiempo de compilación. El tamaño del grupo, según el atributo del kernel, se fija en tiempo de compilación, y luego se puede lanzar el kernel utilizando la sintaxis clásica `<<< , >>>`. Si un kernel utiliza un tamaño de grupo definido en tiempo de compilación, el tamaño del grupo no puede modificarse al lanzar el kernel.

```cpp
// Kernel definition
// Compile time cluster size 2 in X-dimension and 1 in Y and Z dimension
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float* output)
{

}

int main()
{
    float *input, *output;
    // Kernel invocation with compile time cluster size
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    // The grid dimension is not affected by cluster launch, and is still enumerated
    // using number of blocks.
    // The grid dimension must be a multiple of cluster size.
    cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
}
```
