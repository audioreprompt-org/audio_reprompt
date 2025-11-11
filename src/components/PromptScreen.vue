<script setup>
import { ref } from "vue";

// 1. Definición de DEFAULT_PROMPT
const emit = defineEmits(["generate"]);
const DEFAULT_PROMPT =
  "E.g. I'm at the food court and I'm gonna eat a bowl of ramen. The space is noisy and the neon signs are bright.";

const promptText = ref(DEFAULT_PROMPT); // Valor inicial para el mockup
const showError = ref(false);

// 1. Recibe la prop isLoading de App.vue
const props = defineProps({
  isLoading: {
    type: Boolean,
    default: false,
  },
});

const submitPrompt = () => {
  // Limpia el texto para la validación si es el prompt por defecto
  const cleanedPrompt =
    promptText.value.trim() === DEFAULT_PROMPT ? "" : promptText.value.trim();

  if (cleanedPrompt.length > 0) {
    showError.value = false;
    emit("generate", promptText.value.trim());
  } else {
    showError.value = true;
  }
};

// 2. Función para borrar el texto al hacer clic (focus)
const clearPrompt = () => {
  if (promptText.value === DEFAULT_PROMPT) {
    promptText.value = "";
  }
  showError.value = false; // Oculta el error si el usuario intenta escribir
};

// 3. Función para restaurar el texto si el input se queda vacío (blur)
const resetPrompt = () => {
  if (promptText.value.trim() === "") {
    promptText.value = DEFAULT_PROMPT;
  }
};
</script>

<template>
  <div
    class="min-h-screen flex flex-col items-center justify-center p-6 bg-dark-bg text-white relative overflow-hidden"
  >
    <div
      class="absolute inset-0 z-0"
      style="
        /* Ajuste para simular un remolino más pronunciado */
        background-image: radial-gradient(
          circle at 50% 25%,
          /* Centro un poco más arriba */ rgba(167, 139, 250, 0.4) 0%,
          /* light-purple, más claro en el centro */ rgba(126, 34, 206, 0.5) 15%,
          /* main-purple, para la espiral intermedia */ rgba(79, 70, 229, 0.6)
            30%,
          /* main-blue, para la espiral exterior */ rgba(13, 1, 31, 1) 70%
            /* dark-bg, para el fondo oscuro */
        );
        background-size: 150% 150%; /* Aumenta el tamaño para que los colores se distribuyan */
        background-position: 50% 50%; /* Centra el gradiente */
      "
    ></div>

    <div class="relative z-10 max-w-4xl w-full text-center mt-10 space-y-16">
      <h1 class="text-4xl font-extrabold tracking-tight mb-20 text-white">
        Imagine the piece of music to accompany your meal
      </h1>

      <p class="text-xl text-white px-4">
        Where are you, what are you going to eat, and who are you sharing this
        moment with?
      </p>

      <form @submit.prevent="submitPrompt" class="relative px-8">
        <textarea
          v-model="promptText"
          rows="3"
          placeholder=""
          @focus="clearPrompt"
          @blur="resetPrompt"
          class="w-full p-4 border-2 rounded-xl rounded-br-3xl overflow-hidden transition duration-300 shadow-2xl text-sm bg-black/20 text-white border-border-blue-light [text-shadow:_0_0_1px_#000000] placeholder-white/80 focus:ring-4 focus:ring-main-blue/50 focus:border-main-blue/50 resize-y"
          :class="{ 'border-red-500 ring-red-500': showError }"
        ></textarea>
      </form>

      <p v-if="showError" class="text-red-400 text-sm mt-2">
        Please enter a prompt to generate a piece of music.
      </p>

      <button
        @click="submitPrompt"
        :disabled="props.isLoading || showError"
        class="py-4 px-12 text-xl font-bold rounded-full transition duration-300 transform hover:scale-[1.03] shadow-2xl disabled:opacity-50 disabled:cursor-not-allowed"
        style="background: linear-gradient(135deg, #7e22ce 0%, #4f46e5 100%)"
      >
        {{ props.isLoading ? "GENERATING..." : "GENERATE MUSIC" }}
      </button>

      <div
        v-if="props.isLoading"
        class="flex justify-center items-center h-8 space-x-4 mt-4 text-7xl font-bold"
      >
        <span class="text-main-blue animate-pulse" style="animation-delay: 0s">
          &#9835;
        </span>
        <span
          class="text-main-purple animate-pulse"
          style="animation-delay: 0.2s"
        >
          &#9836;
        </span>
        <span
          class="text-light-purple animate-pulse"
          style="animation-delay: 0.4s"
        >
          &#9835;
        </span>
      </div>

      <div class="h-16 flex justify-center items-center">
        <svg
          class="w-3/4 h-full"
          viewBox="0 0 100 20"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <defs>
            <linearGradient id="waveGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stop-color="#7e22ce" />
              <stop offset="50%" stop-color="#4f46e5" />
              <stop offset="100%" stop-color="#7e22ce" />
            </linearGradient>
          </defs>

          <line
            x1="0"
            y1="10"
            x2="100"
            y2="10"
            stroke="#4f46e5"
            stroke-width="0.1"
          />

          <g fill="url(#waveGradient)">
            <rect x="25" y="7.5" width="1" height="5" rx="0.5" />
            <rect x="26.5" y="5" width="1" height="10" rx="0.5" />
            <rect x="28" y="2.5" width="1" height="15" rx="0.5" />
            <rect x="29.5" y="0" width="1" height="20" rx="0.5" />
            <rect x="31" y="2.5" width="1" height="15" rx="0.5" />
            <rect x="32.5" y="5" width="1" height="10" rx="0.5" />
            <rect x="34" y="7.5" width="1" height="5" rx="0.5" />
            <rect x="35.5" y="8.5" width="1" height="3" rx="0.5" />

            <rect x="37" y="1" width="1" height="18" rx="0.5" />
            <rect x="38.5" y="3" width="1" height="14" rx="0.5" />
            <rect x="40" y="5" width="1" height="10" rx="0.5" />
            <rect x="41.5" y="7" width="1" height="6" rx="0.5" />
            <rect x="43" y="9" width="1" height="2" rx="0.5" />

            <rect x="45" y="0" width="1" height="20" rx="0.5" />
            <rect x="46.5" y="1" width="1" height="18" rx="0.5" />
            <rect x="48" y="3" width="1" height="14" rx="0.5" />
            <rect x="49.5" y="5" width="1" height="10" rx="0.5" />
            <rect x="51" y="7" width="1" height="6" rx="0.5" />
            <rect x="52.5" y="9" width="1" height="2" rx="0.5" />

            <rect x="54.5" y="0" width="1" height="20" rx="0.5" />
            <rect x="56" y="1" width="1" height="18" rx="0.5" />
            <rect x="57.5" y="3" width="1" height="14" rx="0.5" />
            <rect x="59" y="5" width="1" height="10" rx="0.5" />
            <rect x="60.5" y="7" width="1" height="6" rx="0.5" />
            <rect x="62" y="9" width="1" height="2" rx="0.5" />

            <rect x="64" y="1" width="1" height="18" rx="0.5" />
            <rect x="65.5" y="3" width="1" height="14" rx="0.5" />
            <rect x="67" y="5" width="1" height="10" rx="0.5" />
            <rect x="68.5" y="7.5" width="1" height="5" rx="0.5" />
          </g>
        </svg>
      </div>
    </div>
  </div>
</template>
