# Script profesional para inicializar CMake (Windows con MSVC Portátil, o Linux/Jetson)
cmakinit() {
    # 1. Limpiar e inicializar la carpeta build
    rm -rf build
    mkdir build && cd build

    # 2. Usar Ninja si está instalado; si no, Makefiles (típico en la Jetson)
    local GEN="Unix Makefiles"
    command -v ninja >/dev/null 2>&1 && GEN="Ninja"

    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*)
            # Windows: MSVC portátil, hay que pasarle rc.exe y mt.exe del SDK
            local SDK_BIN="/c/Users/f78570c/AppData/Local/portable/msvc/msvc-14.44.17.14_sdk-26100/Windows Kits/10/bin/10.0.26100.0/x64"
            cmake .. -G "Ninja" \
              -DCMAKE_CXX_COMPILER="cl.exe" \
              -DCMAKE_RC_COMPILER="${SDK_BIN}/rc.exe" \
              -DCMAKE_MT="${SDK_BIN}/mt.exe"
            ;;
        *)
            # Linux / Jetson: nvcc suele estar en /usr/local/cuda/bin y no en el PATH
            local NVCC="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"
            cmake .. -G "$GEN" \
              -DCMAKE_BUILD_TYPE=Debug \
              -DCMAKE_CUDA_COMPILER="$NVCC"
            ;;
    esac
}

cmakinit
