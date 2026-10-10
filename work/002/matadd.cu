/**
 * Para compilar este ejemplo ejecutar en la línea de comandos:
 * nvcc -arch=native -o matadd matadd.cu
 *
 * Leer arquitectura específica de la máquina:
 * nvidia-smi --query-gpu=compute_cap --format=csv,noheader
 *
 * Forzar arquitectura específica (ejemplo: sm_75 para Turing)
 * nvcc -arch=sm_75 -o matadd matadd.cu
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

// Kernel bidimensional optimizado para suma de matrices
__global__ void matAdd(const float *A, const float *B, float *C, int M, int N)
{
    // Mapear hilos a coordenadas bidimensionales de la matriz
    // x representa las columnas (ancho), y representa las filas (alto)
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    // Control de límites bidimensionales
    if (row < M && col < N)
    {
        // Calcular el índice lineal global (Row-Major Layout)
        int idx = row * N + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main()
{
    std::cout << "Ejemplo de suma de matrices optimizada en CUDA" << std::endl;

    // ========================
    // 1. Dimensiones de la matriz
    // ========================
    int M = 1024; // Número de filas
    int N = 1024; // Número de columnas
    int total_elementos = M * N;
    size_t size = total_elementos * sizeof(float);

    // ========================================
    // 2. Creamos los vectores de las matrices en CPU
    // ========================================
    std::vector<float> h_A(total_elementos);
    std::vector<float> h_B(total_elementos);
    std::vector<float> h_C(total_elementos, 0.0f);

    // ===========================================
    // 3. Generar datos utilizando tus distribuciones
    // ===========================================
    std::random_device rd;
    std::mt19937 gen(rd());

    float a_min = 1.0f;
    float a_max = 2.0f;
    std::uniform_real_distribution<float> d_unif_gen(a_min, a_max);
    for (int i = 0; i < total_elementos; ++i)
    {
        h_A[i] = d_unif_gen(gen);
    }

    float b_mu = 5.0f;
    float b_sigma = 2.5f;
    std::normal_distribution<float> d_norm_gen(b_mu, b_sigma);
    for (int i = 0; i < total_elementos; ++i)
    {
        h_B[i] = d_norm_gen(gen);
    }

    // ============================================
    // 4. Reservar memoria en GPU validando errores
    // ============================================
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    // ===============================================
    // 5. Copiar datos a la GPU (Punteros std::vector)
    // ===============================================
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice));

    // ========================================================
    // 6. Configurar y lanzar Kernel Bidimensional (2D)
    // ========================================================
    std::cout << "Lanzando kernel matAdd..." << std::endl;

    // Configuramos bloques de 16x16 hilos (256 hilos en total por bloque)
    dim3 blockSize(16, 16);

    // Calculamos dinámicamente los bloques necesarios para cubrir M y N
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y);

    matAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N);
    CUDA_CHECK(cudaGetLastError());

    // 6.1. Sincronización explícita de hardware
    CUDA_CHECK(cudaDeviceSynchronize());

    // ===============================================
    // 7. Copiar datos de resultado de vuelta a la CPU
    // ===============================================
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, size, cudaMemcpyDeviceToHost));

    // =============================================
    // 8. Impresión formateada de las primeras 5x5 submatrices
    // =============================================
    std::cout << std::left << std::fixed << std::setprecision(2);
    int filas_print = std::min(M, 5);
    int col_print = std::min(N, 5);

    std::cout << "\nVista parcial de la Matriz Resultado C (Primeras " << filas_print << "x" << col_print << "):\n";
    for (int i = 0; i < filas_print; i++)
    {
        for (int j = 0; j < col_print; j++)
        {
            std::cout << std::setw(8) << h_C[i * N + j] << " ";
        }
        std::cout << (col_print < N ? "... " : "") << std::endl;
    }

    // ============================================
    // 9. Exportación a `suma_matrices.json`
    // ============================================
    std::vector<int> ix(total_elementos);
    std::iota(ix.begin(), ix.end(), 0);

    json datos_json;
    // Metadatos de dimensiones para reconstruir la forma 2D en Python
    datos_json["M"] = M;
    datos_json["N"] = N;

    // Metadatos estadísticos
    datos_json["a_min"] = a_min;
    datos_json["a_max"] = a_max;
    datos_json["b_mu"] = b_mu;
    datos_json["b_sigma"] = b_sigma;

    // Vectores planos consecutivamente ordenados
    datos_json["ix"] = ix;
    datos_json["A"] = h_A;
    datos_json["B"] = h_B;
    datos_json["C"] = h_C;

    std::ofstream archivo("suma_matrices.json");
    if (archivo.is_open())
    {
        archivo << datos_json.dump();
        archivo.close();
        std::cout << "\nJSON generado exitosamente para Jupyter (suma_matrices.json)." << std::endl;
    }
    else
    {
        std::cerr << "Error: No se pudo escribir el archivo JSON." << std::endl;
    }

    // ====================================
    // 10. Limpieza y liberación de memoria
    // ====================================
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
