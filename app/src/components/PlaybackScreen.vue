<script setup>
import { ref } from "vue";

// 1. Definir la prop para recibir los datos de la música
const props = defineProps({
  musicData: {
    type: Object,
    required: true,
    // Define un valor por defecto temporal para evitar errores de plantilla mientras se prueba
    default: () => ({
      original_prompt: "Loading original prompt...",
      improved_prompt: "Loading refined prompt...",
      audio_url: "#",
    }),
  },
});

// 2. Definición del evento que el componente padre escuchará
const emit = defineEmits(["resetAndGoBack"]);

// 3. Estados reactivos y referencias
const isPlaying = ref(false);
const audioPlayer = ref(null);

// 4. Lógica de reproducción/pausa
const togglePlay = () => {
  if (!audioPlayer.value) return;

  if (isPlaying.value) {
    audioPlayer.value.pause();
  } else {
    audioPlayer.value.play().catch((error) => {
      console.error("Error al reproducir el audio:", error);
    });
  }
  isPlaying.value = !isPlaying.value;
};

// 5. Función de descarga
const handleDownload = () => {
  console.log("Iniciando descarga de:", props.musicData.audio_url);
};

// 6. Nueva función para volver y resetear
const goBackAndReset = () => {
  // Asegurarse de pausar el audio al salir
  if (audioPlayer.value && isPlaying.value) {
    audioPlayer.value.pause();
  }
  emit("resetAndGoBack");
};
</script>

<template>
  <div
    class="min-h-screen flex flex-col items-center justify-between p-6 bg-dark-bg text-white relative overflow-hidden"
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

    <div
      class="relative z-10 max-w-4xl w-full text-center mt-24 flex flex-col items-center space-y-16"
    >
      <h1 class="text-4xl font-extrabold tracking-tight mb-8 text-white">
        Our system has associated you with this piece of music
      </h1>

      <div class="w-full flex justify-between space-x-4 px-4 mb-8">
        <div
          class="w-[47%] p-4 bg-white/40 rounded-lg shadow-md border border-white/20 text-left transition duration-300 transform hover:scale-[1.06]"
        >
          <h3 class="text-sm font-bold text-gray-200 mb-1">ORIGINAL PROMPT:</h3>
          <p class="text-xs text-black">
            {{ props.musicData.original_prompt }}
          </p>
        </div>

        <div
          class="w-[47%] p-4 bg-white/40 rounded-lg shadow-md border border-white/20 text-left transition duration-300 transform hover:scale-[1.06]"
        >
          <h3 class="text-sm font-bold text-gray-200 mb-1">REFINED PROMPT:</h3>
          <p class="text-xs text-black">
            {{ props.musicData.improved_prompt }}
          </p>
        </div>
      </div>

      <div class="flex items-center w-full space-x-4 px-8">
        <button
          @click="togglePlay"
          class="w-16 h-16 rounded-full flex items-center justify-center p-3 shadow-2xl transition duration-300"
          style="background: linear-gradient(135deg, #7e22ce 0%, #4f46e5 100%)"
        >
          <svg
            class="w-8 h-8 text-white"
            viewBox="0 0 24 24"
            fill="currentColor"
          >
            <path v-if="!isPlaying" d="M8 5v14l11-7z" />
            <path v-else d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
          </svg>
        </button>

        <div class="flex-grow flex items-center space-x-2">
          <input
            type="range"
            min="0"
            max="100"
            value="30"
            class="w-full h-2 rounded-lg appearance-none cursor-pointer"
          />
          <span class="text-sm text-purple-300 whitespace-nowrap"
            >0:32 / 3:45</span
          >
        </div>
      </div>

      <div class="h-10 flex justify-center items-center mt-4"></div>

      <div class="flex justify-center space-x-12 mt-8">
        <a
          :href="props.musicData.audio_url"
          @click="handleDownload"
          download="generated_music.mp3"
          class="py-4 px-8 text-xl font-bold rounded-full transition duration-300 transform hover:scale-[1.03] shadow-2xl inline-block"
          style="
            background: linear-gradient(135deg, #7e22ce 0%, #4f46e5 100%);
            text-decoration: none;
          "
        >
          DOWNLOAD MUSIC
        </a>

        <button
          @click="goBackAndReset"
          class="py-4 px-8 text-xl font-bold rounded-full transition duration-300 transform hover:scale-[1.03] shadow-2xl"
          style="
            background: linear-gradient(135deg, #4f46e5 0%, #7e22ce 100%);
            text-decoration: none;
          "
        >
          NEW PROMPT
        </button>
      </div>
    </div>
  </div>
  <footer
    class="w-full relative z-10 py-8 bg-black/30 backdrop-blur-sm border-t border-purple-500/20"
  >
    <div class="max-w-4xl mx-auto text-center px-6 text-sm text-purple-400">
      <div class="space-y-2">
        <p class="font-semibold text-purple-300">
          &copy; 2025 Audio Reprompting Project
        </p>
        <p class="text-purple-400">Universidad de Los Andes</p>
      </div>

      <div class="mt-6">
        <p class="text-purple-300 font-medium mb-2">Developers:</p>
        <div
          class="flex flex-nowrap justify-center gap-x-8 text-purple-400 whitespace-nowrap px-4"
        >
          <span>Jose Daniel Garcia Davila</span>
          <span>Juana Valentina Mendoza Santamaría</span>
          <span>Valentina Nariño Chicaguy</span>
          <span>Jorge Luis Sarmiento Herrera</span>
          <span>Mario Fernando Reyes Ojeda</span>
        </div>
      </div>
    </div>
  </footer>
</template>
