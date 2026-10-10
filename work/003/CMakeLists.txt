cmake_minimum_required(VERSION 3.18)

# Elegir la arquitectura de GPU según la máquina donde se compila.
# Tiene que ir ANTES de project(), porque project() pone un default propio.
# Se puede forzar a mano con -DCMAKE_CUDA_ARCHITECTURES=XX.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
        set(CMAKE_CUDA_ARCHITECTURES 87)   # Jetson Orin Nano (sm_87)
    else()
        set(CMAKE_CUDA_ARCHITECTURES 75)   # GPU de la PC Windows (sm_75)
    endif()
endif()

project(CudaMediaMovil LANGUAGES CXX CUDA)

# Configurar de forma profesional el estándar de C++ y CUDA
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Windows: linkear el runtime de C/C++ (CRT) de forma estática (/MT, /MTd en Debug)
# para que el .exe no dependa de ucrtbased.dll / vcruntime140d.dll.
# En Linux esta variable se ignora. El runtime de CUDA (cudart) ya se linkea
# estático por defecto en las dos plataformas.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# Definir el ejecutable y sus archivos fuentes
add_executable(mediamovil mediamovil.cu)

# Configurar propiedades profesionales para el ejecutable en Windows
set_target_properties(mediamovil PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    OUTPUT_NAME "mediamovil"
)

##################################

# Cómo usar:

# Cuando pases al siguiente ejercicio de CUDA:
# 1. Creás tu carpeta y metés tu código junto con el archivo CMakeLists.txt.
# 2. Desde la raíz de ese nuevo ejercicio, escribís simplemente:bash
# ```bash
# ./cmakinit.sh
# ```

# (Eso te crea la carpeta, te mete adentro y te configura el proyecto solito).
# 3. Para compilar cada cambio que hagas a partir de ahí, solo tirás:bash
# ```bash
# cmake --build .
# ```