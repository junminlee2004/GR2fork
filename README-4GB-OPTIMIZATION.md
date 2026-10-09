# GR2fork - Low VRAM 4GB Optimization

Este fork de GR2fork incluye optimizaciones para PCs con **4GB de VRAM**, aplicando las mejores opciones de Shadlix y ajustes adicionales para evitar crashes por `OutOfDeviceMemory`.

## Cambios Aplicados

### 1. Garbage Collection Optimizado para ≤4GB VRAM
**Archivo:** `src/video_core/texture_cache/texture_cache.cpp`

Cuando el sistema detecta ≤4GB de VRAM, se aplican umbrales de GC más agresivos:
- `pressure_gc_memory`: 50% del VRAM total
- `critical_gc_memory`: 75% del VRAM total
- `trigger_gc_memory`: 25% del VRAM total

Esto libera texturas antes de alcanzar el límite y previene crashes.

### 2. DEFAULT_CRITICAL_GC_MEMORY Reducido
**Archivo:** `src/video_core/texture_cache/texture_cache.h`

- **Antes:** `3_GB`
- **Ahora:** `2_GB`

Para GPUs que no reportan uso de memoria, el GC crítico se activa antes.

### 3. Readbacks Mode: Relaxed
**Archivo:** `src/core/emulator_settings.h`

- **Antes:** `GpuReadbacksMode::Disabled`
- **Ahora:** `GpuReadbacksMode::Relaxed`

Modo Relaxed de Shadlix: equilibria compatibilidad y rendimiento. El modo Disabled causa problemas de iluminación y partículas faltantes; Precise gasta más VRAM.

### 4. FSR Habilitado por Defecto
**Archivo:** `src/core/emulator_settings.h`

- **Antes:** `fsr_enabled{false}`
- **Ahora:** `fsr_enabled{true}`

Renderiza a menor resolución interna (720p) y escala con FSR. Reduce drásticamente el uso de VRAM.

### 5. Resolución 720p por Defecto
**Archivo:** `src/core/emulator_settings.h`

- **Antes:** `window_width{1280}` / `window_height{720}` (sin comentario)
- **Ahora:** Optimizado explícitamente para 720p

### 6. Vulkan Validation Core Desactivado
**Archivo:** `src/core/emulator_settings.h`

- **Antes:** `vkvalidation_core_enabled{true}`
- **Ahora:** `vkvalidation_core_enabled{false}`

La validación de Vulkan añade overhead innecesario en GPUs con poca VRAM.

## Opciones Adicionales de Shadlix Recomendadas

Estas opciones **ya están incluidas** en la base de Shadlix/GR2fork y se recomiendan para 4GB VRAM:

| Opción | Dónde | Valor Recomendado | Efecto |
|--------|-------|-------------------|--------|
| **RCAS Sharpening** | Settings → Graphics | ON, attenuation 0.250 | Mejora nitidez al escalar con FSR |
| **Safe Tiling** | Settings → Experimental | ON | Optimiza uso de VRAM con texturas |
| **isDevKit** | Settings → Experimental | OFF | Evita reservar 8000MB de RAM |
| **Neo Mode (PS4 Pro)** | Settings → Experimental | OFF | Más recursos para el juego |
| **Pipeline Cache** | Settings → Vulkan | ON | Reduce stutter y recargas de shaders |
| **isDevKit** | config.toml | `false` | Verificar que esté desactivado |

## Compilación

### Requisitos
- Visual Studio 2022 (v17.10+)
- CMake 3.24+
- Vulkan SDK 1.3+
- Windows 10/11, Linux, o macOS

### Pasos
```bash
# Clonar el fork
git clone https://github.com/Gabriel0066/GR2fork.git
cd GR2fork

# Cambiar a la rama optimizada
git checkout low-vram-4gb

# Compilar (Windows)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# O en Linux
cmake -B build -DCMAKE_BUILD_TYPE=Release -G Ninja
ninja -C build
```

## Uso

1. Compila el emulador (o descarga un release)
2. Coloca los dumps de tus juegos en una carpeta
3. Abre `shadPS4QtLauncher.exe`
4. Configura el directorio de juegos
5. Asegúrate de que `config.toml` tenga `isDevKit = false`
6. En Settings → Graphics: activa FSR + RCAS
7. En Settings → Experimental: activa Safe Tiling
8. En Settings → Vulkan: activa Pipeline Cache

## Configuración Recomendada (config.toml)

```toml
[General]
isDevKit = false
isNeo = false

[Graphics]
fsrEnabled = true
rcasEnabled = true
rcasAttenuation = 250
windowWidth = 1280
windowHeight = 720

[Validation]
validationEnabled = false
validationCoreEnabled = false

[Vulkan]
pipelineCacheEnable = true
```

## Notas

- **4GB VRAM es el mínimo** para shadPS4. Algunos juegos pueden no correr bien o requerir patches específicos.
- Si experimentas crashes, cierra aplicaciones que consuman VRAM (navegador, Discord, etc.)
- Los patches de juego (Lowest Model Detail, No Motion Blur, etc.) ayudan significativamente.
- Actualiza los drivers de tu GPU regularmente.

## Referencias

- [Shadlix (diegolix29)](https://github.com/diegolix29/shadPS4) — Fork base con las opciones de optimización
- [GR2fork (junminlee2004)](https://github.com/junminlee2004/GR2fork) — Fork base para Gravity Rush 2
- [shadPS4 Official](https://github.com/shadps4-emu/shadPS4) — Emulador base
