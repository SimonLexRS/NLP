/**
 * Compartir la página en redes sociales (PROYECTO-3).
 *
 * Los controles viven en el footer de index.html (.footer-share); este módulo
 * completa los href de cada red con la URL actual y el texto de la página,
 * activa el botón nativo si el navegador soporta navigator.share y gestiona
 * la copia al portapapeles con su feedback (role="status" en el HTML).
 */

const TEXTO_COMPARTIR =
  "Clasificador de Noticias: pipeline NLP (categoría, tono y sentimiento) 100% en el navegador";

/** URL limpia de la página (sin fragmento) para compartir. */
function urlPagina() {
  return window.location.href.split("#")[0];
}

/**
 * Completa los href de los enlaces de intención de cada red social.
 * Los href base ya están en el HTML; aquí se les añaden los parámetros.
 */
function completarEnlacesRedes() {
  const url = encodeURIComponent(urlPagina());
  const texto = encodeURIComponent(TEXTO_COMPARTIR);
  const hrefs = {
    "share-x": `https://twitter.com/intent/tweet?text=${texto}&url=${url}`,
    "share-whatsapp": `https://wa.me/?text=${texto}%20${url}`,
    "share-linkedin": `https://www.linkedin.com/sharing/share-offsite/?url=${url}`,
    "share-telegram": `https://t.me/share/url?url=${url}&text=${texto}`,
  };
  for (const [id, href] of Object.entries(hrefs)) {
    const enlace = document.getElementById(id);
    if (enlace) enlace.href = href;
  }
}

/** Muestra un mensaje breve en la sección de compartir y lo limpia a los ~2,5 s. */
function mostrarEstadoCompartir(msg) {
  const estado = document.getElementById("share-status");
  if (!estado) return;
  estado.textContent = msg;
  window.clearTimeout(mostrarEstadoCompartir._timer);
  mostrarEstadoCompartir._timer = window.setTimeout(() => {
    estado.textContent = "";
  }, 2500);
}

/** Copia al portapapeles con fallback para contextos no seguros (http://…). */
async function copiarAlPortapapeles(url) {
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(url);
      return true;
    }
  } catch (e) {
    /* sigue al fallback */
  }
  try {
    const aux = document.createElement("textarea");
    aux.value = url;
    aux.setAttribute("readonly", "");
    aux.style.position = "fixed";
    aux.style.opacity = "0";
    document.body.appendChild(aux);
    aux.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(aux);
    return ok;
  } catch (e) {
    return false;
  }
}

/** Activa el botón nativo (Web Share API) solo si el navegador lo soporta. */
function activarBotonNativo(url, titulo) {
  const btn = document.getElementById("btn-share-nativo");
  if (!btn || typeof navigator.share !== "function") return;
  btn.style.display = "";
  btn.addEventListener("click", async () => {
    try {
      await navigator.share({ title: titulo, text: TEXTO_COMPARTIR, url });
    } catch (e) {
      // El usuario canceló el diálogo nativo: no es un error.
    }
  });
}

/**
 * Inicializa la sección de compartir del footer.
 * Idempotente: se llama una sola vez desde app.js (DOMContentLoaded).
 */
export function inicializarCompartir() {
  const url = urlPagina();
  const titulo = document.title;

  completarEnlacesRedes();
  activarBotonNativo(url, titulo);

  const btnCopiar = document.getElementById("btn-copiar-enlace");
  if (btnCopiar) {
    btnCopiar.addEventListener("click", async () => {
      const ok = await copiarAlPortapapeles(url);
      mostrarEstadoCompartir(ok ? "Enlace copiado ✓" : "No se pudo copiar el enlace");
    });
  }
}