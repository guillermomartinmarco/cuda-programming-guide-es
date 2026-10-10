/**
 * Leer `mejoras.md` para entender cómo está optimizado este código
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

// Kernel optimizado: Cada hilo calcula
// una muestra 't' para un 'periodo' dado
__global__ void mediaMovilOptimizada(const float *y,
                                     float *out_matrix, const int *periodos,
                                     int N, int num_periodos)
{

    // Mapear el hilo a una muestra de tiempo (X) y a un índice de período (Y)
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_periodo = blockIdx.y * blockDim.y + threadIdx.y;

    // Control de límites
    if (t >= N || idx_periodo >= num_periodos)
        return;

    int periodo = periodos[idx_periodo];

    // Matriz de salida en un solo puntero plano para conservar coalescencia
    // Fila: idx_periodo, Columna: t
    size_t out_idx = idx_periodo * N + t;

    if (t < periodo - 1)
    {
        out_matrix[out_idx] = 0.0f;
    }
    else
    {
        float suma = 0.0f;
        // Aunque sigue habiendo un bucle, todos los hilos en el mismo eje Y
        // tienen el MISMO periodo, eliminando el 100% de la warp divergence en esa fila.
        for (int i = 0; i < periodo; i++)
        {
            suma += y[t - i];
        }
        out_matrix[out_idx] = suma / (float)periodo;
    }
}

// Función `main`
int main()
{
    std::cout << "Ejemplo de media móvil concurrente en CUDA" << std::endl;
    int N = 1048576; // Número de elementos del vector de señal

    // ================================================
    // 1. Crear el vector de señal a procesar en la CPU
    //    y el vector de períodos a pasarle a la GPU
    // ================================================
    std::vector<float> h_y(N); // Vector de la señal
    std::vector<int> periodos = {2, 3, 4, 5, 6, 7, 8, 9, 10};
    int nPeriodos = periodos.size(); // Cantidad de periodos

    // =========================================================
    // 2. Generar datos de la señal de prueba: vector `y`
    //    Generador de números aleatorios con hardware o semilla
    // =========================================================
    std::random_device rd;
    std::mt19937 gen(rd());
    // Definir el rango [min, max] (por ejemplo, entre 0 y 100)
    std::uniform_real_distribution<float> distribucion(0.0f, 1.0f);
    for (int i = 0; i < N; ++i)
    {
        float numero_aleatorio = distribucion(gen);
        h_y[i] = numero_aleatorio;
    }

    // ============================================
    // 3. Reservar memoria en GPU validando errores
    // ============================================

    // Vector de datos de la señal
    float *d_y;
    size_t sz_y = N * sizeof(float);
    CUDA_CHECK(cudaMalloc((void **)&d_y, sz_y));

    // Matriz de los resultados
    float *d_out;
    size_t sz_out = nPeriodos * N * sizeof(float);
    CUDA_CHECK(cudaMalloc((void **)&d_out, sz_out));

    // Vector de periodos
    int *d_periodos;
    size_t sz_periodos = nPeriodos * sizeof(int);
    CUDA_CHECK(cudaMalloc((void **)&d_periodos, sz_periodos));

    // ===============================================
    // 4. Copiar a la GPU. Convertir el vector del
    //    host `h_y` de std::vector a un puntero crudo
    // ===============================================
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), sz_y, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_periodos, periodos.data(), sz_periodos, cudaMemcpyHostToDevice));

    // ================
    // 5. Lanzar Kernel
    // ================
    std::cout << "Lanzando kernel mediaMovil..." << std::endl;
    // 256 hilos para el eje del tiempo, 1 para periodos
    dim3 blockSize(256, 1);

    // `nPeriodos` bloques en el eje Y (uno por periodo)
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, nPeriodos);

    mediaMovilOptimizada<<<gridSize, blockSize>>>(d_y, d_out,
                                                  d_periodos, N,
                                                  nPeriodos);

    CUDA_CHECK(cudaGetLastError());
    // 5.1. Espera a que terminen todos los hilos del kernel
    CUDA_CHECK(cudaDeviceSynchronize());

    // ============================================================
    // 6. Copiar de vuelta a la CPU (Actualizado para matriz plana)
    // ============================================================
    // Vector plano para recibir todo junto
    std::vector<float> h_out(nPeriodos * N);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, sz_out, cudaMemcpyDeviceToHost));

    // =====================================================
    // 8. Exportación a `datos.json` (Optimizado y dinámico)
    // =====================================================
    std::vector<int> ix(N);

    // Llenar con valores de 0 a N-1
    std::iota(ix.begin(), ix.end(), 0);

    json datos_json;
    datos_json["ix"] = ix;
    datos_json["y"] = h_y; // Señal original

    // Extraer dinámicamente cada bloque de tamaño N desde el vector plano h_out
    for (int i = 0; i < nPeriodos; ++i)
    {
        std::string nombre_columna = "media_movil_p" + std::to_string(periodos[i]);

        // Calculamos dónde empieza y dónde termina el bloque de este período
        auto inicio_bloque = h_out.begin() + (i * N);
        auto fin_bloque = inicio_bloque + N;

        // Creamos un sub-vector temporal para este período específico
        std::vector<float> mm_periodo(inicio_bloque, fin_bloque);

        // Lo asignamos directamente al JSON
        datos_json[nombre_columna] = mm_periodo;
    }

    // Exportar a archivo compacto para conservar espacio
    std::ofstream archivo("datos.json");
    if (archivo.is_open())
    {
        archivo << datos_json.dump();
        archivo.close();
        std::cout << "JSON generado exitosamente para Jupyter con la nueva matriz optimizada." << std::endl;
    }

    // ===================================
    // 9. Limpieza y liberación de memoria
    // ===================================
    cudaFree(d_y);
    cudaFree(d_out);
    cudaFree(d_periodos);

    return 0;
}
