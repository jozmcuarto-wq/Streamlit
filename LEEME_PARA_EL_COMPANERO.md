# Entrega para la pareja

El proyecto de Machine Learning ya está preparado en esta carpeta:

- Clasificador supervisado de calidad de vino con datos públicos UCI.
- `train.py`, `model.joblib`, `app.py`, dataset, dependencias y Dockerfile.
- README con ejecución local, Docker y despliegue en Compute Engine.
- Checklist y evidencia de validación local.

## Lo que falta completar en pareja

1. **Autores:** editar la línea de autores en `README.md`.
2. **GitHub:** basta con que una persona de la pareja cree/use una cuenta. No hace falta instalar Git: creen un repositorio en la web, descompriman `ml_streamlit_project.zip`, entren al repo, pulsen **Add file → Upload files**, arrastren los archivos del proyecto y confirmen el commit. Verifiquen que `Dockerfile`, `model.joblib`, `.dockerignore` y la carpeta `data` quedaron en la raíz. [Instrucciones oficiales para crear cuenta](https://docs.github.com/en/account-and-profile/how-tos/account-management/creating-an-account-on-github) y [subir archivos desde el navegador](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository). No suban tokens ni contraseñas.
3. **GCP:** usar el proyecto/cuenta autorizado por el profesor. Antes de crear la VM, revisar facturación y costos. La guía de despliegue está en `README.md`.
4. **Docker/GCP:** el build y la app ya se probaron en el computador local. Para la entrega de GCP todavía hay que instalar Docker en la VM, clonar el repo, construir la imagen allí, iniciar el contenedor y permitir TCP 8501 para la etiqueta de red `streamlit-server`.
5. **Evidencias:** guardar la IP pública y una captura de la app funcionando; completar `EVIDENCIA_PRUEBA_CRUZADA.md` cuando otra pareja haya reproducido el repo.

## Comandos centrales en la VM

```bash
sudo apt update
sudo apt install -y docker.io git
sudo systemctl enable --now docker
git clone <URL_DEL_REPOSITORIO>
cd <CARPETA_DEL_REPOSITORIO>
docker build -t ml-streamlit-vino .
docker run -d --name ml-streamlit-vino --restart unless-stopped -p 8501:8501 ml-streamlit-vino
docker ps
docker logs ml-streamlit-vino
```

La regla de firewall y las comprobaciones están detalladas en `README.md`. Si el profesor entrega créditos o una VM institucional, usen esa opción para evitar cargos personales.

En el computador de la estudiante no hay cuenta de GitHub ni Git instalado. Docker Desktop 29.9.3 sí quedó instalado; la imagen se construyó y el contenedor está funcionando localmente en `http://localhost:8501`. La cuenta/repo de GitHub, la VM de GCP, la IP pública y la prueba cruzada siguen pendientes.

## Mensaje breve para enviar

> Ya preparé el proyecto de Machine Learning y te mando el ZIP con el código, el modelo, los datos y las instrucciones. También instalé Docker, construí la imagen y probé la app localmente. Para cerrar la entrega falta crear/subir el repo de GitHub, desplegarla en una VM de GCP con la cuenta o créditos autorizados por el profesor, probar la IP pública y hacer la prueba cruzada. El README tiene los comandos y el orden.
