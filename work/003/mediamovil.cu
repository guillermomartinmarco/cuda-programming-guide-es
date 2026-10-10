/**
 * En este ejemplo demostraremos el uso de intrínsecos de CUDA.
 * 1. Crearemos un vector de 1024 datos aleatorios entre 0.0 y 1.0.
 * 2. crearemos 8 kernels.
 * 3. Calcularemos la media móvil de ese vector con cada kernel.
 * 4. La particularidad es que cada kernel calculará la media móvil sobre los
 *    mismos datos, pero en diferentes periodos.
 * 5. Cada hilo devolverá un vector de la media móvil correspondiente que
 *    calculó. Las dimensiones de ese vector serán las mismas que el vector de
 *    datos, es decir, 1024 elementos.
 * 6. El hilo 1, calculará la media movil de periodo 2,
 *    El hilo 2, calculará la media movil de periodo 4,
 *    El hilo 3, calculará la media movil de periodo 8,
 *    El hilo 4, calculará la media movil de periodo 16,
 *    El hilo 5, calculará la media movil de periodo 32,
 *    El hilo 6, calculará la media movil de periodo 64,
 *    El hilo 7, calculará la media movil de periodo 128,
 *    El hilo 8, calculará la media movil de periodo 256.
 * 7. Para simplificar y sin escalar el ejemplo, el kernel completo
 *    devolverá cada una de los 8 vectores (señales) de medias móviles
 *    independientemente.
 * 8. Estos datos se guardarán en un archivo JSON para su posterior análisis
 *    con la notebook de jupyter del ejemplo, con el siguiente formato:
 * {
 *  "ix":  [0, 1, 2, ..., 1023], // índice de los datos
 *  "y":   [0.45, 0.12, 0.89, ...], // Señal original
 *  "media_movil_p2": [0.45, 0.28, 0.50, ...], // media móvil de periodo 2
 *  "media_movil_p4": [0.45, 0.35, 0.42, ...], // media móvil de periodo 4
 *  "...": [],
 *  "media_movil_p256": [0.45, 0.50, 0.55, ...]  // media móvil de periodo 256
 * }
 *
 * Ejemplos de compilación:
 * ------------------------
 * Para compilar este simple ejemplo ejecutar en la línea de comandos:
 * nvcc -o mediamovil mediamovil.cu
 * 
 * Forzar arquitectura nativa
 * nvcc -arch=native -o mediamovil mediamovil.cu
 * 
 * Leer arquitectura especifica de la máqiona
 * nvidia-smi --query-gpu=compute_cap --format=csv,noheader
 * 7.5
 * 
 * Forzar arquitectura específica (ejemplo: sm_75 para Turing)
 * nvcc -arch=sm_75 -o mediamovil mediamovil.cu
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

// Para std::iota
#include <numeric>

// Incluir la biblioteca JSON de nlohmann
#include "../include/nlohmann/json.hpp"

#include <random>
#include <cuda_runtime.h>

using json = nlohmann::json;

// Manejador de errores de CUDA básico
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
      if (abort) exit(code);
   }
}

// Kernel para calcular las 8 medias móviles
// utilizando un hilo por cada periodo
__global__ void mediaMovil( float *y,
                            float *mm1,
                            float *mm2,
                            float *mm3,
                            float *mm4,
                            float *mm5,
                            float *mm6,
                            float *mm7,
                            float *mm8,
                            int N) {
    
    // El ID del hilo (del 0 al 7) determina
    // qué media móvil calcula este hilo
    int id_mm = threadIdx.x; 

    // Variables locales para el puntero
    // de salida y el periodo de este hilo
    float* out_ptr = nullptr;
    int periodo = 0;

    // Asignación explícita (no escalable, "a mano") usando un switch
    switch(id_mm) {
        case 0: out_ptr = mm1; periodo = 2;   break;
        case 1: out_ptr = mm2; periodo = 4;   break;
        case 2: out_ptr = mm3; periodo = 8;   break;
        case 3: out_ptr = mm4; periodo = 16;  break;
        case 4: out_ptr = mm5; periodo = 32;  break;
        case 5: out_ptr = mm6; periodo = 64;  break;
        case 6: out_ptr = mm7; periodo = 128; break;
        case 7: out_ptr = mm8; periodo = 256; break;
        default: return; // Por seguridad si se lanzan más de 8 hilos
    }

    // Cada hilo recorre de forma independiente
    // TODO el vector 'y' de tamaño `N` y calcula su media móvil
    for (int t = 0; t < N; t++) {
        
        // Si no hay suficientes elementos
        // históricos previos, se llena con 0.0f
        if (t < periodo - 1) {
            out_ptr[t] = 0.0f;
        } 
        else {
            // Calcular la suma acumulada para
            // el periodo correspondiente
            float suma = 0.0f;
            for (int i = 0; i < periodo; i++) {
                suma += y[t - i];
            }
            // Guardar el promedio en el vector
            // de salida asignado a este hilo
            out_ptr[t] = suma / (float)periodo;
        }
    }
}


// Función `main`
int main() {
    std::cout << "Ejemplo de media móvil concurrente en CUDA" << std::endl;
    int N = 1024; // número de elementos en cada vector
    size_t size = N * sizeof(float);

    // 1. Crear los vectores de datos en la CPU
    std::vector<float> h_y(N); // Vector de la señal
    std::vector<float> h_mm1(N); // Vector de la media móvil de periodo 2
    std::vector<float> h_mm2(N); // Vector de la media móvil de periodo 4
    std::vector<float> h_mm3(N); // Vector de la media móvil de periodo 8
    std::vector<float> h_mm4(N); // Vector de la media móvil de periodo 16
    std::vector<float> h_mm5(N); // Vector de la media móvil de periodo 32
    std::vector<float> h_mm6(N); // Vector de la media móvil de periodo 64
    std::vector<float> h_mm7(N); // Vector de la media móvil de periodo 128
    std::vector<float> h_mm8(N); // Vector de la media móvil de periodo 256

    // 2. Generar datos de la señal de prueba: vector `y`
    //    Generador de números aleatorios con hardware o semilla
    std::random_device rd;
    std::mt19937 gen(rd());
    // Definir el rango [min, max] (por ejemplo, entre 0 y 100)
    std::uniform_real_distribution<float> distribucion(0.0f, 1.0f);
    for (int i = 0; i < N; ++i) {
        float numero_aleatorio = distribucion(gen);
        h_y[i] = numero_aleatorio;
    }

    // 3. Reservar memoria en GPU validando errores
    float *d_y, *d_mm1, *d_mm2, *d_mm3, *d_mm4, *d_mm5, *d_mm6, *d_mm7, *d_mm8;
    CUDA_CHECK(cudaMalloc((void**)&d_y, size));
    CUDA_CHECK(cudaMalloc((void**)&d_mm1, size));
    CUDA_CHECK(cudaMalloc((void**)&d_mm2, size));
    CUDA_CHECK(cudaMalloc((void**)&d_mm3, size));
    CUDA_CHECK(cudaMalloc((void**)&d_mm4, size));
    CUDA_CHECK(cudaMalloc((void**)&d_mm5, size));
    CUDA_CHECK(cudaMalloc((void**)&d_mm6, size));
    CUDA_CHECK(cudaMalloc((void**)&d_mm7, size));
    CUDA_CHECK(cudaMalloc((void**)&d_mm8, size));

    // 4. Copiar a la GPU. Convertir el vector del
    //    host `h_y` de std::vector a un puntero crudo
    CUDA_CHECK(cudaMemcpy(d_y,     h_y.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mm1, h_mm1.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mm2, h_mm2.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mm3, h_mm3.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mm4, h_mm4.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mm5, h_mm5.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mm6, h_mm6.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mm7, h_mm7.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mm8, h_mm8.data(), size, cudaMemcpyHostToDevice));

    // 5. Lanzar Kernel
    std::cout << "Lanzando kernel mediaMovil..." << std::endl;
    mediaMovil<<<1, 8>>>(d_y, d_mm1, d_mm2, d_mm3, d_mm4, d_mm5, d_mm6, d_mm7, d_mm8, N);
    CUDA_CHECK(cudaGetLastError());
    
    // 5.1. Espera a que terminen todos los hilos del kernel
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6. Copiar de vuelta a la CPU. Convertir los datos float
    //    crudos a vectores float de std::vector
    CUDA_CHECK(cudaMemcpy(h_mm1.data(), d_mm1, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mm2.data(), d_mm2, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mm3.data(), d_mm3, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mm4.data(), d_mm4, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mm5.data(), d_mm5, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mm6.data(), d_mm6, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mm7.data(), d_mm7, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_mm8.data(), d_mm8, size, cudaMemcpyDeviceToHost));

    // 8. Exportación todo a `datos.json` usando la librería nlohmann/json
    std::vector<int> ix(N); // Vector de índices
    std::iota(ix.begin(), ix.end(), 0); // Llenar con valores de 0 a N-1
    // 8.1. Crear los 8 vectores con las medias móviles devueltas por el kernel
    std::vector<float> mm1(h_mm1.begin(), h_mm1.end());
    std::vector<float> mm2(h_mm2.begin(), h_mm2.end());
    std::vector<float> mm3(h_mm3.begin(), h_mm3.end());
    std::vector<float> mm4(h_mm4.begin(), h_mm4.end());
    std::vector<float> mm5(h_mm5.begin(), h_mm5.end());
    std::vector<float> mm6(h_mm6.begin(), h_mm6.end());
    std::vector<float> mm7(h_mm7.begin(), h_mm7.end());
    std::vector<float> mm8(h_mm8.begin(), h_mm8.end());
    // 8.2. Crear el vector con los periodos que usamos en cada hilo
    //      (ej: 2, 4, 8, 16, 32, 64, 128, 256)
    std::vector<int> periodos = {2, 4, 8, 16, 32, 64, 128, 256};
    // 8.3. Construir el objeto JSON
    json datos_json;
    // 8.4. Guardamos el índice y la señal original
    datos_json["ix"] = ix;
    datos_json["y"]  = h_y;
    // 8.5. Guardamos dinámicamente las 8 medias móviles
    //      usando su periodo como nombre de la llave
    for (int i = 0; i < 8; ++i) {
        std::string nombre_columna = "media_movil_p" + std::to_string(periodos[i]);
        switch (i){
        case 0:
            datos_json[nombre_columna] = mm1;
            break;
        case 1:
            datos_json[nombre_columna] = mm2;
            break;
        case 2:
            datos_json[nombre_columna] = mm3;
            break;
        case 3:
            datos_json[nombre_columna] = mm4;
            break;
        case 4:
            datos_json[nombre_columna] = mm5;
            break;
        case 5:
            datos_json[nombre_columna] = mm6;
            break;
        case 6:
            datos_json[nombre_columna] = mm7;
            break;
        case 7:
            datos_json[nombre_columna] = mm8;
            break;
        default:
            break;
        }
    }
    // 8.6. Exportar a archivo sin indentación (para
    //      que ocupe menos espacio y guarde rápido)
    std::ofstream archivo("datos.json");
    if (archivo.is_open()) {
        // Sin argumentos para guardarlo compacto
        archivo << datos_json.dump();
        archivo.close();
        std::cout << "JSON generado exitosamente para Jupyter." << std::endl;
    }

    // 9. Limpieza y liberación de memoria
    cudaFree(d_y);
    cudaFree(d_mm1);
    cudaFree(d_mm2);
    cudaFree(d_mm3);
    cudaFree(d_mm4);
    cudaFree(d_mm5);
    cudaFree(d_mm6);
    cudaFree(d_mm7);
    cudaFree(d_mm8);

    return 0;
}
