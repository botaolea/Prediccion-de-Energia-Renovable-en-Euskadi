# 🚀 Guía rápida para Visual Studio Code

Esta guía resuelve los errores más comunes al ejecutar el proyecto en VS Code.

---

## ⚡ Inicio rápido (3 pasos)

### 1️⃣ Descomprime el ZIP

Descomprime `proyecto_energia_euskadi.zip` en una carpeta, por ejemplo:
- **Windows**: `C:\Users\TuUsuario\proyecto_energia_euskadi`
- **Mac/Linux**: `/home/tuusuario/proyecto_energia_euskadi`

### 2️⃣ Abre la carpeta en VS Code

```bash
cd proyecto_energia_euskadi
code .
```

> ⚠️ **MUY IMPORTANTE**: Abre la **carpeta raíz** del proyecto, NO un archivo suelto. Si abres un archivo individual, los imports fallarán.

### 3️⃣ Ejecuta el script de configuración

Abre una terminal en VS Code (`Ctrl+ñ` o `Terminal → New Terminal`) y ejecuta:

```bash
# Linux/Mac
bash setup_env.sh

# Windows (PowerShell)
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python diagnostico.py
```

El script:
1. Crea un entorno virtual `venv/`
2. Instala todas las dependencias
3. Ejecuta el diagnóstico automático

---

## 🐧 Requisitos específicos para Linux (Ubuntu/Debian)

Instalar Python, pip y el entorno virtual:

```bash
sudo apt update

sudo apt install -y \
python3 \
python3-pip \
python3-venv \
python3-dev \
build-essential \
git
```

Comprobar la instalación:

```bash
python3 --version
pip3 --version
```

Se recomienda Python **3.10 o 3.11**.

Si utilizas Python 3.12, asegúrate de que todas las dependencias son compatibles.

## 🐛 Errores comunes y soluciones

### Error 1: `ModuleNotFoundError: No module named 'streamlit'`

**Causa**: No has activado el entorno virtual o no has instalado las dependencias.

**Solución**:
```bash
# Activa el venv
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Instala dependencias
pip install -r requirements.txt
```

### Error 2: `ModuleNotFoundError: No module named 'app'` o `'src'`

**Causa**: Estás ejecutando un archivo desde fuera de la raíz del proyecto.

**Solución**: Asegúrate de que el CWD (current working directory) es la raíz del proyecto:
```bash
# Verifica dónde estás
pwd   # Linux/Mac
cd    # Windows

# Debes estar en:
# /ruta/a/proyecto_energia_euskadi
# y ver: app.py, config.yaml, src/, app/, etc.
```

### Error 3: VS Code usa el Python incorrecto

**Causa**: VS Code no tiene seleccionado el intérprete del venv.

**Solución**:
1. Pulsa `Ctrl+Shift+P` (Windows/Linux) o `Cmd+Shift+P` (Mac)
2. Escribe `Python: Select Interpreter`
3. Selecciona: `./venv/bin/python` (o `.\venv\Scripts\python.exe` en Windows)

### Error 4: `streamlit: command not found`

**Causa**: El comando `streamlit` no está en el PATH.

**Solución**: Usa el módulo de Python en su lugar:
```bash
python -m streamlit run app.py
```

### Error 5: `FileNotFoundError: config.yaml`

**Causa**: El script no encuentra `config.yaml` porque se ejecuta desde otra carpeta.

**Solución**: Ejecuta SIEMPRE desde la raíz:
```bash
cd /ruta/a/proyecto_energia_euskadi
python scripts/run_pipeline.py
```

### Error 6: `streamlit.runtime.scriptrunner_utils.script_run_context.ScriptRunContext` warnings

**Causa**: Estás importando módulos de Streamlit fuera del runtime.

**Solución**: Es solo un warning, **se puede ignorar**. No afecta a la app.

### Error 7: Las páginas no aparecen en la navegación

**Causa**: Versión de Streamlit < 1.36 (no soporta `st.Page`).

**Solución**:
```bash
pip install --upgrade streamlit
```

### Error 8: SHAP tarda mucho o da error de memoria

**Causa**: El cálculo SHAP sobre muchos datos es costoso.

**Solución**: Ya está limitado a 1000 muestras. Si aún así falla, reduce en `src/model_trainer.py`:
```python
self.compute_shap(X_train_scaled, max_samples=200)  # Reducir de 1000 a 200
```

### Error 9: Folium / streamlit-folium no carga el mapa

**Causa**: Versión incompatible de streamlit-folium.

**Solución**:
```bash
pip install --upgrade folium streamlit-folium
```

### Error 10: `PermissionError` escribiendo en `data/` o `models/`

**Causa**: Permisos del sistema de archivos.

**Solución**:
```bash
# Linux/Mac
chmod -R 755 data/ models/

# Windows: ejecuta VS Code como administrador (no recomendado)
# Mejor: cambia la propiedad de la carpeta
```

---

## 🎯 Cómo ejecutar en VS Code (recomendado)

### Opción A: Con F5 (Debugger) — RECOMENDADA

Ya tienes `.vscode/launch.json` configurado. Solo pulsa **F5** y elige:
- **"Streamlit: app.py"** → lanza la app web
- **"Pipeline completo"** → reentrena el modelo
- **"Tests (pytest)"** → ejecuta los tests

### Opción B: Desde la terminal integrada

1. Abre terminal: `Ctrl+ñ`
2. Activa venv:
   ```bash
   # Linux/Mac
   source venv/bin/activate
   # Windows
   venv\Scripts\activate
   ```
3. Ejecuta:
   ```bash
   streamlit run app.py
   ```

### Opción C: Desde la paleta de comandos

1. `Ctrl+Shift+P`
2. Escribe `Python: Run File in Terminal`
3. (Pero antes configura el intérprete correcto)

---

## 📂 Estructura esperada

Tu VS Code debería verse así en el explorador:

```
📁 proyecto_energia_euskadi/
├── 📁 .vscode/
│   ├── launch.json         ← Configuración F5
│   └── settings.json       ← Configuración Python
├── 📁 .streamlit/
│   ├── secrets.toml        ← Tu API key (no subir a git)
│   └── secrets.toml.example
├── 📁 app/
│   ├── 📁 components/
│   └── 📁 pages/
├── 📁 data/
│   ├── 📁 raw/
│   └── 📁 processed/
├── 📁 models/
├── 📁 src/
├── 📁 tests/
├── 📁 scripts/
├── 📄 app.py               ← APP PRINCIPAL
├── 📄 config.yaml
├── 📄 requirements.txt
├── 📄 diagnostico.py       ← Ejecuta si algo falla
├── 📄 setup_env.sh         ← Setup inicial
└── 📄 README.md
```

---

## 🧪 Verificar que todo funciona

```bash
# 1. Ejecuta el diagnóstico
python diagnostico.py

# 2. Ejecuta los tests
python -m pytest tests/ -v

# 3. Lanza la app
streamlit run app.py
```

Si los 3 comandos terminan sin errores, **el proyecto está listo**.

---

## 🌐 ¿Subir a GitHub o a otro sitio?

### NO es necesario subirlo para que funcione

La app **funciona 100% en local** en tu VS Code. No necesitas subirla a ningún sitio para usarla.

### ¿Cuándo sí subirlo?

| Caso | Dónde |
|------|-------|
| **Compartir código** | GitHub (público/privado) |
| **App online accesible 24/7** | Streamlit Cloud (gratis) |
| **App online con más recursos** | Hugging Face Spaces, Railway, Render |
| **Colaborar con otras persona** | GitHub |

### Pasos para subir a GitHub (opcional)

1. Crea un repositorio en https://github.com/new
   - Nombre: `proyecto_energia_euskadi`
   - Público o privado
   - NO inicializar con README (ya lo tienes)

2. En VS Code, abre la terminal y ejecuta:
   ```bash
   git remote add origin https://github.com/TU_USUARIO/proyecto_energia_euskadi.git
   git branch -M main
   git push -u origin main
   ```

3. Para desplegar en **Streamlit Cloud** (gratis, online):
   - Ve a https://share.streamlit.io/
   - Conecta tu cuenta de GitHub
   - Selecciona el repo `TU_USUARIO/proyecto_energia_euskadi`
   - Main file path: `app.py`
   - Click en "Deploy"
   - ¡Listo! Tu app estará en `https://tu-usuario-proyecto-energia-euskadi.streamlit.app`

---

## ❓ ¿Sigues con errores?

Ejecuta el diagnóstico y envíame la salida:

```bash
python diagnostico.py > diagnostico_salida.txt 2>&1
```

Luego comparte el archivo `diagnostico_salida.txt`.

---

## 📋 Checklist final

- [ ] Python 3.10+ instalado
- [ ] VS Code instalado con extensión Python
- [ ] Carpeta del proyecto descomprimida
- [ ] `bash setup_env.sh` ejecutado (o pasos manuales)
- [ ] `python diagnostico.py` muestra "TODO OK"
- [ ] Intérprete de VS Code seleccionado (`venv/bin/python`)
- [ ] `streamlit run app.py` funciona (HTTP 200)
- [ ] Navegación entre páginas funciona (5 páginas)
- [ ] Tests pasan: `pytest tests/ -v`
