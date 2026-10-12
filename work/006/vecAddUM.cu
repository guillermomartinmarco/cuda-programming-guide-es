/**
 * Este ejemplo suma dos vectores de señales de una cantidad determinada
 * de elementos. Cada suma es ejecutada en un thread, por lo tanto sumará
 * concurrentemente cada elemento de los dos vectores de entrada.
 * Realizaremos la suma vectorial siguiente:
 *
 *               C = A + B
 *
 * Además de esto realizaremos esta misma suma en la CPU.
 *
 * La particularidad de este ejercicio es que usaremos los mismos
 * punteros de memoria de los vectores sumandos `A` y `B`.
 * Haremos uso de la memoria unificada.
 *
 * Por último realizaremos una comparación de resultados GPU vs CPU.
 */

// ========
// INCLUDES
// ========
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
#include <cuda/cmath> // para usar cuda::ceil_div

// ===================================
// Manejador de errores de CUDA básico
// ===================================
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

// ===========
// KERNEL CUDA
// ===========
__global__ void vecAdd(float *v1, float *v2, float *res, int dimVec)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < dimVec)
    {
        res[i] = v1[i] + v2[i];
    }
}

// ====================================================================
// Generador de un vector de datos aleatorios con distrinución uniforme
// ====================================================================
int genVecDistUnif(float *vector, int dimVector,
                   float min, float max)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> generador(min, max);
    for (int i = 0; i < dimVector; ++i)
        vector[i] = generador(gen);
    return 0;
}

// ==================================================================
// Generador de un vector de datos aleatorios con distrinución normal
// ==================================================================
int genVecDistNorm(float *vector, int dimVector,
                   float mu, float sigma)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> generador(mu, sigma);
    for (int i = 0; i < dimVector; ++i)
        vector[i] = generador(gen);
    return 0;
}

// =================================
// Suma de vectores en el host (CPU)
// =================================
int serialVecAdd(float *v1, float *v2, float *res, int dimVec)
{
    for (int i = 0; i < dimVec; i++)
        res[i] = v1[i] + v2[i];
    return 0;
}

// ================================================
// Comparador de vectores con Épsilon de tolerancia
// ================================================
bool vectorApproximatelyEqual(float *v1, float *v2, int dimVec)
{
    // Definimos una tolerancia para el error de redondeo flotante
    const float epsilon = 1e-5f; 
    
    for (int i = 0; i < dimVec; i++)
    {
        // Calculamos la diferencia absoluta entre CPU y GPU
        float diferencia = std::abs(v1[i] - v2[i]);
        if (diferencia > epsilon)
        {
            // Imprimir el primer error encontrado ayuda muchísimo a debuggear
            std::cout << "Discrepancia en indice " << i 
                      << " | GPU: " << v1[i] << " | CPU: " << v2[i] 
                      << " | Dif: " << diferencia << std::endl;
            return false;
        }
    }
    return true;
}

// =============
// main del host
// =============
int main()
{
    std::cout << "Ejemplo de suma de vectores en CUDA" << std::endl;

    // =============================
    // 1. Parámetros de los vectores
    // =============================

    // Cantidad de elementos de cada señal
    int N = 10000000;
    size_t size = N * sizeof(float);

    // =====================================
    // 2. Punteros a los vectores de memoria
    // =====================================
    float *A = nullptr;
    float *B = nullptr;
    float *C = nullptr;
    float *comparisonResult = (float *)malloc(size);

    // =============================================
    // 3. Usa memoria unificada para asignar búferes
    // =============================================
    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    // ==================================
    // 4. Inicializar vectores en el host
    // ==================================
    float A_min = 1.0f;
    float A_max = 2.0f;
    float B_mu = 2.0f;
    float B_sigma = 0.5f;
    genVecDistUnif(A, N, A_min, A_max);
    genVecDistNorm(B, N, B_mu, B_sigma);

    // ===================================================
    // 5. Inicia el kernel. La memoria unificada
    //    garantizará que la GPU pueda acceder a A, B y C.
    // ===================================================
    int threads = 256;
    int blocks = cuda::ceil_div(N, threads);
    vecAdd<<<blocks, threads>>>(A, B, C, N);
    // Esperar a que el kernel termine de ejecutarse
    cudaDeviceSynchronize();

    // =========================================================
    // 6. Realizar el cálculo en serie en la CPU para compararlo
    // =========================================================
    serialVecAdd(A, B, comparisonResult, N);

    // =================================================================
    // 7. Verifica que la CPU y la GPU hayan obtenido el mismo resultado
    // =================================================================
    if (vectorApproximatelyEqual(C, comparisonResult, N))
    {
        printf("Memoria unificada: las respuestas de la CPU y la GPU coinciden\n");
    }
    else
    {
        printf("Memoria unificada: Error: las respuestas de la CPU y la GPU no coinciden\n");
    }

    // ==============================
    // 8. Imprimir algunos resultados
    // ==============================
    // Alinea el texto a la izquierda dentro de su espacio
    std::cout << std::left;
    // Cabecera
    std::cout << std::setw(10) << "i"
              << std::setw(10) << "A"
              << std::setw(10) << "B"
              << std::setw(10) << "C(GPU)"
              << std::setw(10) << "C(CPU)" << "\n";
    // Precisión de los valores
    std::cout << std::fixed << std::setprecision(2);
    for (int i = 0; i < 10; i++)
    {
        std::cout << std::setw(10) << i
                  << std::setw(10) << A[i]
                  << std::setw(10) << B[i]
                  << std::setw(10) << C[i]
                  << std::setw(10) << comparisonResult[i] << "\n";
    }

    // =============================
    // 9. Exportación a `datos.json`
    // =============================
    std::vector<int> ix(N);
    // Llenar con valores de 0 a N-1
    std::iota(ix.begin(), ix.end(), 0);

    json datos_json;

    // Guardamos los metadatos de configuración de las señales
    datos_json["A_min"] = A_min;
    datos_json["A_max"] = A_max;
    datos_json["B_mu"] = B_mu;
    datos_json["B_sigma"] = B_sigma;

    datos_json["ix"] = ix;
    // Vector `A` señal uniforme
    std::vector<float> vA(A, A + N);
    datos_json["A"] = vA;
    // Vector `B` señal normal
    std::vector<float> vB(B, B + N);
    datos_json["B"] = vB;
    // Vector Resultado `C` de la suma vectorial en GPU
    std::vector<float> vC_GPU(C, C + N);
    datos_json["C_GPU"] = vC_GPU;
    // Vector Resultado `C` de la suma vectorial en CPU
    std::vector<float> vC_CPU(comparisonResult, comparisonResult + N);
    datos_json["C_CPU"] = vC_CPU;

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
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    free(comparisonResult);

    return 0;
}
