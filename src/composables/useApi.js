import { ref } from "vue";

// Define la URL base de la API (se inyectará en tiempo de compilación/Docker)
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000/api";

export function useApi() {
  const data = ref(null);
  const error = ref(null);
  const loading = ref(false);
  const generateMusic = async (prompt) => {
    loading.value = true;
    error.value = null;
    try {
      const response = await fetch(`${API_BASE_URL}/audio/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      if (!response.ok) throw new Error("Error al generar la música.");
      data.value = await response.json();
      data.value.original_prompt = prompt;

    } catch (err) {
      error.value = err.message;
    } finally {
      loading.value = false;
    }
  };

  return { data, error, loading, generateMusic };
}
