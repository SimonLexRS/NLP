// theme.js — Botón de modo claro/oscuro.
// El atributo data-theme en <html> lo fija el script inline de <head> antes
// del primer paint (anti-FOUC); este módulo solo gestiona el botón, la
// persistencia y el <meta name="theme-color">. Toda lectura/escritura de
// localStorage está protegida: el tema nunca debe romper la herramienta.

const THEME_KEY = "pln-theme";
const META_COLOR = { light: "#f4f6f8", dark: "#0b1220" };

/** Devuelve "dark"/"light" si el valor es válido; null en otro caso. */
function normalizar(valor) {
  return valor === "dark" || valor === "light" ? valor : null;
}

function actualizarMeta(theme) {
  const meta = document.getElementById("meta-theme-color");
  if (meta) meta.setAttribute("content", META_COLOR[theme]);
}

export function getTheme() {
  return normalizar(document.documentElement.getAttribute("data-theme")) || "light";
}

/**
 * Aplica un tema al documento y sincroniza el botón y el meta theme-color.
 * @param {string} next "dark" | "light" (cualquier otro valor cae en "light").
 * @param {{persist?: boolean}} [opciones] persist=false cuando la fuente es
 *   el sistema o el arranque (no conviene guardar una elección no elegida).
 */
export function setTheme(next, { persist = true } = {}) {
  const theme = normalizar(next) || "light";
  document.documentElement.setAttribute("data-theme", theme);

  const btn = document.getElementById("btn-tema");
  if (btn) {
    const oscuro = theme === "dark";
    btn.setAttribute("aria-pressed", oscuro ? "true" : "false");
    btn.setAttribute("aria-label", oscuro ? "Activar modo claro" : "Activar modo oscuro");
    btn.setAttribute("data-tema-icono", oscuro ? "moon" : "sun");
  }
  actualizarMeta(theme);

  if (persist) {
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch (_) {
      /* Almacenamiento bloqueado (modo privado/iframe): el tema sigue vivo en memoria. */
    }
  }
}

function temaGuardado() {
  try {
    return normalizar(localStorage.getItem(THEME_KEY));
  } catch (_) {
    return null;
  }
}

/** Sigue al sistema solo mientras el usuario no haya elegido un tema. */
function observarSistema() {
  if (!window.matchMedia) return;
  const mq = window.matchMedia("(prefers-color-scheme: dark)");
  const handler = (evento) => {
    if (!temaGuardado()) setTheme(evento.matches ? "dark" : "light", { persist: false });
  };
  if (typeof mq.addEventListener === "function") mq.addEventListener("change", handler);
  else if (typeof mq.addListener === "function") mq.addListener(handler);
}

/** Reconcilia el DOM con el atributo ya puesto por el script inline. Idempotente. */
export function initTheme() {
  const guardado = temaGuardado();
  if (!guardado && window.matchMedia) {
    setTheme(window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light", {
      persist: false,
    });
  } else {
    setTheme(guardado || document.documentElement.getAttribute("data-theme") || "light", {
      persist: false,
    });
  }

  const btn = document.getElementById("btn-tema");
  if (btn && !btn.dataset.temaListo) {
    btn.dataset.temaListo = "true";
    btn.addEventListener("click", () => setTheme(getTheme() === "dark" ? "light" : "dark"));
  }
  observarSistema();
}