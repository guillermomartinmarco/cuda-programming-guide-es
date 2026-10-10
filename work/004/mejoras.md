# Mejoras del código de la media móvil

En el código anterior, [`../003/mediamovil.cu`](../003/mediamovil.cu) estamos aplicando exactamente la lógica de **Paralelismo por Estrategia**, es decir, que cada hilo calcule una media móvil distinta. El hecho de pasar un único vector `d_y` para que todos los hilos lo lean concurrentemente es conceptualmente el camino correcto para optimizar hiperparámetros.

Sin embargo, analizando el código detalladamente, tenemos un **problema crítico de rendimiento y arquitectura** que está haciendo que la GPU trabaje de forma extremadamente ineficiente (de hecho, este código corre más lento que en un solo núcleo de CPU).

A continuación, la explicación de los problemas graves de la implementación anteriormente citada y cómo reestructurarla para aprovechar realmente la GPU.

## Los 3 problemas críticos del código anterior

### 1. Warp Divergence masiva (El bucle interno dinámico)

En el kernel tenemos un bucle anidado que depende directamente del período de cada hilo:

```cpp
for (int i = 0; i < periodo; i++) {
    suma += y[t - i];
}
```

- **Qué pasa en la GPU:** El hilo 0 (período 2) hace 2 iteraciones. El hilo 7 (período 256) hace 256 iteraciones. Como los hilos de un mismo _warp_ (grupo de 32 hilos) se ejecutan obligatoriamente en estricto paso sincronizado, **el hilo 0 tendrá que quedarse congelado (esperando sin hacer nada) durante 254 ciclos** mientras el hilo 7 termina de iterar. El paralelismo se destruye por completo.

### 2. Subutilización extrema del Hardware

Estamos lanzando el kernel con la configuración `<<<1, 8>>>`.

- Estamos usando **un solo bloque y solo 8 hilos**.

- Una GPU moderna maneja miles de hilos en paralelo. Al lanzar 8 hilos, estamos usando aproximadamente el **0.01%** de la capacidad de cómputo de un solo Multiprocesador de Streaming (SM) de la tarjeta, dejando el resto de la GPU completamente ociosa.

### 3. Cero Coalescencia de Escritura y paso de punteros ineficiente

Pasar 8 punteros independientes por parámetro (`mm1`, `mm2`... `mm8`) rompe cualquier estructura escalable. Además, cada hilo escribe de forma secuencial en su propio vector asignado, lo que destruye la **coalescencia de memoria** al escribir los resultados de vuelta.

---

## La Solución Correcta: Invertir los Ejes del Kernel

Para solucionar la _warp divergence_ y saturar la GPU, tenemos que cambiar la jerarquía de hilos. En lugar de que _un hilo calcule todo el tiempo de un período_, debemos hacer que **cada hilo calcule una única muestra de tiempo para un período específico**.

## Código Sugerido (Kernel Optimizado Escalable)

En este diseño, la dimensión **X** del bloque se encarga del tiempo (las muestras) y la dimensión **Y** se encarga de los diferentes períodos (los hiperparámetros). De este modo, los hilos de un mismo warp ejecutan exactamente las mismas instrucciones al mismo tiempo.

```cpp
// Kernel optimizado: Cada hilo calcula una muestra 't' para un 'periodo' dado
__global__ void mediaMovilOptimizada(const float *y, float *out_matrix, const int *periodos, int N, int num_periodos) {
    // Mapear el hilo a una muestra de tiempo (X) y a un índice de período (Y)
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_periodo = blockIdx.y * blockDim.y + threadIdx.y;

    // Control de límites
    if (t >= N || idx_periodo >= num_periodos) return;

    int periodo = periodos[idx_periodo];

    // Matriz de salida en un solo puntero plano para conservar coalescencia
    // Fila: idx_periodo, Columna: t
    size_t out_idx = idx_periodo * N + t;

    if (t < periodo - 1) {
        out_matrix[out_idx] = 0.0f;
    } else {
        float suma = 0.0f;
        // Aunque sigue habiendo un bucle, todos los hilos en el mismo eje Y
        // tienen el MISMO periodo, eliminando el 100% de la warp divergence en esa fila.
        for (int i = 0; i < periodo; i++) {
            suma += y[t - i];
        }
        out_matrix[out_idx] = suma / (float)periodo;
    }
}
```

## ¿Cómo cambia el Host (`main`) con este enfoque?

1. **Memoria unificada para salidas:** En lugar de crear 8 vectores y 8 punteros en la GPU (`d_mm1`, `d_mm2`, etc.), reservamos un solo bloque plano de memoria que contenga todas las señales juntas (una matriz de `num_periodos * N`).

   ```cpp
   // 8 periodos por 1024 datos
   float *d_out;size_t total_size = 8 * N * sizeof(float);
   cudaMalloc((void**)&d_out, total_size);
   ```

2. **Pasar los períodos como un vector a la GPU:** Creamos un pequeño vector con los hiperparámetros (`{2, 4, 8, 16, 32, 64, 128, 256}`), lo copiamos a la GPU (`d_periodos`) y se lo pasamos al kernel para eliminar el bloque `switch`.

3. **Lanzamiento masivo de hilos:** Ahora configuramos la grilla para lanzar miles de hilos distribuidos.

   ```cpp
   // 256 hilos para el eje del tiempo, 1 para periodos
   dim3 blockSize(256, 1);

   // 8 bloques en el eje Y (uno por periodo)
   dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 8);

   mediaMovilOptimizada<<<gridSize, blockSize>>>(d_y,
                                                 d_out,
                                                 d_periodos,
                                                 N,
                                                 8);
   ```

Con este cambio de arquitectura, la GPU procesará las 8.192 operaciones (1024 muestras × 8 períodos) de forma simultánea en microsegundos, respetando la coalescencia de lectura y erradicando la penalización por la divergencia de hilos.

## Adaptar la generación del JSON

En esta sección adaptaremos la parte del **guardado en el JSON** para que pueda leer esta nueva estructura de matriz plana de forma limpia

Para adaptar el guardado en el archivo JSON a la nueva estructura de matriz plana (donde todos los resultados están juntos en el vector `h_out`), **debemos modificar el código del host en C++**.

Como la GPU nos devuelve un único vector lineal que contiene las 8 medias móviles una detrás de otra (primero los 1024 datos de P2, luego los 1024 de P4, etc.), podemos usar **aritmética de punteros o iteradores** para extraer cada segmento dinámicamente.

Aquí tenemos el reemplazo limpio para la **sección 8** de la función `main`:

```cpp
// =========================================================================
// 6. Copiar de vuelta a la CPU (Actualizado para matriz plana)
// =========================================================================
int num_periodos = 8;

// Vector plano para recibir todo junto
std::vector<float> h_out(num_periodos * N);
CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, num_periodos * size, cudaMemcpyDeviceToHost));

// =========================================================================
// 8. Exportación a `datos.json` (Optimizado y dinámico)
// =========================================================================
std::vector<int> ix(N); 

// Llenar con valores de 0 a N-1
std::iota(ix.begin(), ix.end(), 0);

std::vector<int> periodos = {2, 4, 8, 16, 32, 64, 128, 256};

json datos_json;
datos_json["ix"] = ix;
datos_json["y"]  = h_y; // Señal original

// Extraer dinámicamente cada bloque de tamaño N desde el vector plano h_out
for (int i = 0; i < num_periodos; ++i) {
    std::string nombre_columna = "media_movil_p" + std::to_string(periodos[i]);
    
    // Calculamos dónde empieza y dónde termina el bloque de este período
    auto inicio_bloque = h_out.begin() + (i * N);
    auto fin_bloque    = inicio_bloque + N;
    
    // Creamos un sub-vector temporal para este período específico
    std::vector<float> mm_periodo(inicio_bloque, fin_bloque);
    
    // Lo asignamos directamente al JSON
    datos_json[nombre_columna] = mm_periodo;
}

// Exportar a archivo compacto para conservar espacio
std::ofstream archivo("datos.json");
if (archivo.is_open()) {
    archivo << datos_json.dump();
    archivo.close();
    std::cout << "JSON generado exitosamente para Jupyter con la nueva matriz optimizada." << std::endl;
}
```

## ¿Por qué este cambio es mejor para el proyecto?

- **Elimina el `switch-case`:** Ya no dependemos de estructuras rígidas. Si en el futuro queremos probar con 50 períodos de medias móviles diferentes, por ejemplo para optimización en trading, solo tenemos que agregar los números al vector `periodos` y cambiar el valor de `num_periodos`. El JSON se armará solo.

- **Consumo de memoria eficiente:** Al usar el constructor por rango (`std::vector<float>(inicio, fin)`), el código copia directamente los segmentos correctos de la memoria consecutiva de manera sumamente rápida.

> [!WARNING]
> No olvidemos cambiar también la sección de liberación de memoria al final del `main` reemplazando los 8 `cudaFree` individuales por un único `cudaFree(d_out);`.
