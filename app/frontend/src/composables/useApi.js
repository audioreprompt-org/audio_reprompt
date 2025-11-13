import { ref } from "vue";

// Define la URL base de la API (se inyectará en tiempo de compilación/Docker)
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

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

/**
export function useApi() {
  const data = ref(null);
  const error = ref(null);
  const loading = ref(false);
  const generateMusic = async (prompt) => {
    loading.value = true;
    error.value = null;
    try {
      // Simula la llamada al backend de FastAPI

      const mockData = {
        original_prompt: prompt,
        audio_id: "1",
        improved_prompt:
          "We are at a bustling farmers market on a crisp autumn morning. The air is cool and smells faintly of fresh herbs and baked bread. Sunlight filters through rows of colorful stalls piled high with pumpkins, berries, and flowers. Nearby, a musician strums a gentle tune. I pick up a Honeycrisp apple its skin glossy red and gold and take a bite; it is juicy, sweet, and slightly tart, echoing the freshness of the season.",
        audio_url:
          "https://drive.google.com/file/d/1dedO8MrnKvC3eh2MN5uouLx7U46f3PtV/view?usp=sharing",
      };

      // 2. Simular un tiempo de latencia de 3 segundos (para ver el progreso)
      await new Promise((resolve) => setTimeout(resolve, 30000));

      // 3. Asignar datos simulados
      data.value = mockData;

      /*
      const response = await fetch(`${API_BASE_URL}/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      if (!response.ok) throw new Error("Error al generar la música.");
      data.value = await response.json();
      */
/**
    } catch (err) {
      error.value = err.message;
    } finally {
      loading.value = false;
    }
  };

  return { data, error, loading, generateMusic };
}
*/
