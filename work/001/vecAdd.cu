/**
 * Este ejemplo suma dos vectores de señales de una cantidad determinada
 * de elementos. Cada suma es ejecutada en un thread, por lo tanto sumará
 * concurrentemente cada elemento de los dos vectores de entrada.
 * Realizaremos la suma vectorial siguiente:
 *
 *               z = x + y
 *
 * Para compilar este simple ejemplo ejecutar en la línea de comandos:
 * nvcc -o vecAdd vecAdd.cu
 *
 * Forzar arquitectura nativa
 * nvcc -arch=native -o vecAdd vecAdd.cu
 *
 * Leer arquitectura especifica de la máqiona
 * nvidia-smi --query-gpu=compute_cap --format=csv,noheader
 * 7.5
 *
 * Forzar arquitectura específica (ejemplo: sm_75 para Turing)
 * nvcc -arch=sm_75 -o vecAdd vecAdd.cu
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>

// Para std::iota
#include <numeric>

// Incluir la biblioteca JSON de nlohmann
#include "../include/nlohmann/json.hpp"
using json = nlohmann::json;

#include <random>
#include <cuda_runtime.h>

// Manejador de errores de CUDA básico
#define CUDA_CHECK(ans)                       \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort)
            exit(code);
    }
}

__global__ void vecAdd(float *x, float *y, float *z, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        z[i] = x[i] + y[i];
    }
}

int main()
{
    std::cout << "Ejemplo de suma de vectores en CUDA" << std::endl;

    // ======================
    // 1. Creamos las señales
    // ======================
    int N = 1048576; // Cantidad de elementos de cada señal
    size_t size = N * sizeof(float);

    // ======================================
    // 2. Creamos los vectores de las señales
    // ======================================
    std::vector<float> h_x(N); // Vector de la señal `x`
    std::vector<float> h_y(N); // Vector de la señal `y`
    std::vector<float> h_z(N); // Vector del resultado `z`

    // ===============================
    // 3. Generar datos de las señales
    // ===============================
    std::random_device rd;
    std::mt19937 gen(rd());
    // 3.1. Generar señal `x` en una distribución uniforme
    //      valor minimo: -1.0
    //      valor máximo:  1.0
    float x_min = 1.0f;
    float x_max = 2.0f;
    std::uniform_real_distribution<float> d_unif_gen(x_min, x_max);
    for (int i = 0; i < N; ++i)
        h_x[i] = d_unif_gen(gen);
    // 3.2. Generar señal `y` en una distribución normal
    //      media:                2.0
    //      desviación estándar:  0.5
    float y_mu = 5.0f;
    float y_sigma = 0.5f;
    std::normal_distribution<float> d_norm_gen(y_mu, y_sigma);
    for (int i = 0; i < N; ++i)
        h_y[i] = d_norm_gen(gen);

    // ============================================
    // 4. Reservar memoria en GPU validando errores
    // ============================================
    float *d_x, *d_y, *d_z;
    CUDA_CHECK(cudaMalloc((void **)&d_x, size));
    CUDA_CHECK(cudaMalloc((void **)&d_y, size));
    CUDA_CHECK(cudaMalloc((void **)&d_z, size));

    // ========================================
    // 5. Copiar datos a la GPU.
    //    Convertir vectores del host `h_`
    //    desde `std::vector` a punteros crudos
    // ========================================
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), size, cudaMemcpyHostToDevice));

    // ================
    // 6. Lanzar Kernel
    // ================
    std::cout << "Lanzando kernel vecAdd..." << std::endl;
    // ATENCIÓN: Un hilo por cada elemento
    int threadsPerBlock = 256; // Máximo de threads por bloque
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_x, d_y, d_z, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ===============================================
    // 7. Copiar datos de resultado de vuelta a la CPU
    // ===============================================
    CUDA_CHECK(cudaMemcpy(h_z.data(), d_z, size, cudaMemcpyDeviceToHost));

    // ==============================
    // 8. Imprimir algunos resultados
    // ==============================
    // Alinea el texto a la izquierda dentro de su espacio
    std::cout << std::left;
    // Cabecera
    std::cout << std::setw(5) << "i" << std::setw(10) << "h_x[i]" << std::setw(10) << "h_y[i]" << std::setw(10) << "h_z[i]" << "\n";
    // Precisión de los valores
    std::cout << std::fixed << std::setprecision(2);
    for (int i = 0; i < 10; i++)
    {
        std::cout << std::setw(5) << i << std::setw(10) << h_x[i] << std::setw(10) << h_y[i] << std::setw(10) << h_z[i] << "\n";
    }

    // =============================
    // 9. Exportación a `datos.json`
    // =============================
    std::vector<int> ix(N);
    // Llenar con valores de 0 a N-1
    std::iota(ix.begin(), ix.end(), 0);
    
    json datos_json;
    
    // Guardamos los metadatos de configuración de las señales
    datos_json["x_min"] = x_min;
    datos_json["x_max"] = x_max;
    datos_json["y_mu"]   = y_mu;
    datos_json["y_sigma"]= y_sigma;

    datos_json["ix"] = ix;
    datos_json["x"] = h_x; // Vector `x` señal uniforme
    datos_json["y"] = h_y; // Vector `y` señal normal
    datos_json["z"] = h_z; // Vector Resultado `z` de la suma vectorial

    // Abrimos el flujo de archivo para escribir el JSON en disco
    std::ofstream archivo("suma_vectores.json");
    if (archivo.is_open())
    {
        // dump() sin argumentos genera el archivo de forma compacta (sin espacios ni saltos de línea)
        archivo << datos_json.dump();
        archivo.close();
        std::cout << "JSON generado exitosamente para Jupyter (suma_vectores.json)." << std::endl;
    }
    else
    {
        std::cerr << "Error: No se pudo crear o abrir el archivo JSON para escritura." << std::endl;
    }

    // ====================================
    // 10. Limpieza y liberación de memoria
    // ====================================
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);

    return 0;
}
