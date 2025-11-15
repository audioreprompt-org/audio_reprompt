<script setup>
import { ref, onMounted, watch } from "vue";

const props = defineProps({
  /**
   * 1. Define prop to receive generated music data from the parent component.
   *    This includes the original and improved prompts, and the generated audio file URL.
   */
  musicData: {
    type: Object,
    required: true,
    default: () => ({
      original_prompt: "Loading original prompt...",
      improved_prompt: "Loading refined prompt...",
      audio_base64: "",
    }),
  },
});

// 2. Define event to notify the parent component when the user wants to reset and return to the previous screen.
const emit = defineEmits(["resetAndGoBack"]);

// 3. Reactive states and references.
const isPlaying = ref(false);
const audioPlayer = ref(null);

const duration = ref(0);
const current = ref(0);
const percent = ref(0);
const displayTime = (s) => {
  /**
   * Helper function to format time from seconds to "MM:SS".
   * @param {number} s - Time in seconds.
   * @returns {string} - Formatted time string.
   */
  if (!Number.isFinite(s)) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60)
    .toString()
    .padStart(2, "0");
  return `${m}:${sec}`;
};

const audioSrc = ref("");

const makeDataUrl = (b64) => {
  return `data:audio/wav;base64,${b64}`;
};

onMounted(() => {
  /**
   * onMounted: Set the initial audio URL when the component loads.
   */
  audioSrc.value = makeDataUrl(props.musicData.audio_base64);
});

watch(
  /**
   * Watcher: Reset the audio player when the received audio base64 changes.
   */
  () => props.musicData.audio_base64,
  (b64) => {
    audioSrc.value = makeDataUrl(b64);
    if (audioPlayer.value) {
      isPlaying.value = false;
      audioPlayer.value.pause();
      audioPlayer.value.load();
      duration.value = 0;
      current.value = 0;
      percent.value = 0;
    }
  }
);

const togglePlay = async () => {
  /**
   * Toggles between play and pause states for the audio player.
   */
  const el = audioPlayer.value;
  if (!el) return;
  try {
    if (isPlaying.value) {
      el.pause();
      isPlaying.value = false;
    } else {
      await el.play();
      isPlaying.value = true;
    }
  } catch (err) {
    console.error("Error al reproducir el audio:", err);
  }
};

const onLoadedMetadata = () => {
  /**
   * Audio event: Called when metadata (like duration) is available.
   */
  duration.value = audioPlayer.value?.duration ?? 0;
};

const onTimeUpdate = () => {
  /**
   * Audio event: Updates the current playback time and progress percentage.
   */
  current.value = audioPlayer.value?.currentTime ?? 0;
  if (duration.value) {
    percent.value = (current.value / duration.value) * 100;
  }
};

const onEnded = () => {
  /**
   * Audio event: Triggered when playback finishes.
   */
  isPlaying.value = false;
};

const base64ToBlob = (b64) => {
  const byteCharacters = atob(b64);
  const byteNumbers = new Array(byteCharacters.length);
  for (let i = 0; i < byteCharacters.length; i++) {
    byteNumbers[i] = byteCharacters.charCodeAt(i);
  }
  const byteArray = new Uint8Array(byteNumbers);
  return new Blob([byteArray], { type: "audio/wav" });
};

const handleDownload = async () => {
  /**
   * 5. Function to handle music file download.
   * Fetches the audio blob and triggers a local download in the browser.
   */
  const b64 = props.musicData.audio_base64;
  const blob = base64ToBlob(b64);
  const url = URL.createObjectURL(blob);
  try {
    const a = document.createElement("a");
    a.href = url;
    // choose extension that matches your Content-Type (audio/wav, audio/mpeg, etc.)
    a.download = "generated_music.wav";
    document.body.appendChild(a);
    a.click();
    a.remove();
  } catch (e) {
    console.error("Fallo la descarga:", e);
  } finally {
    URL.revokeObjectURL(url);
  }
};

const goBackAndReset = () => {
  /**
   * 6. Function to pause audio (if playing) and notify parent to reset UI.
   */
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
          /* Centro un poco más arriba */ rgba(88, 34, 250, 0.2) 0%,
          /* light-purple, más claro en el centro */ rgba(126, 34, 206, 0.5) 15%,
          /* main-purple, para la espiral intermedia */ rgba(17, 4, 152, 0.6)
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
      <h1
        class="text-4xl font-extrabold tracking-tight mb-8 text-white [text-shadow:_0_0_4px_rgba(168,85,247,0.35),_0_0_8px_rgba(168,85,247,0.25)]"
      >
        Our system has associated you with this piece of music
      </h1>

      <div class="w-full flex justify-between space-x-4 px-4 mb-8">
        <div
          class="w-[47%] p-4 bg-black/40 rounded-lg shadow-md border border-white/20 text-left cursor-pointer transition duration-300 transform hover:scale-[1.09]"
        >
          <h3 class="text-sm font-bold text-gray-200 mb-1">ORIGINAL PROMPT:</h3>
          <p class="text-xs text-white">
            {{ props.musicData.original_prompt }}
          </p>
        </div>

        <div
          class="w-[47%] p-4 bg-black/40 rounded-lg shadow-md border border-white/20 text-left cursor-pointer transition duration-300 transform hover:scale-[1.09]"
        >
          <h3 class="text-sm font-bold text-gray-200 mb-1">REFINED PROMPT:</h3>
          <p class="text-xs text-white">
            {{ props.musicData.improved_prompt }}
          </p>
        </div>
      </div>

      <audio
        ref="audioPlayer"
        :src="audioSrc"
        preload="metadata"
        crossorigin="anonymous"
        @loadedmetadata="onLoadedMetadata"
        @timeupdate="onTimeUpdate"
        @ended="onEnded"
      ></audio>

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
            :value="percent"
            @input="onSeek"
            class="w-full h-2 rounded-lg appearance-none cursor-pointer"
          />
          <span class="text-sm text-purple-300 whitespace-nowrap">
            {{ displayTime(current) }} / {{ displayTime(duration) }}
          </span>
        </div>
      </div>

      <div class="flex justify-center space-x-12 mt-8">
        <a
          href="#"
          @click.prevent="handleDownload"
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
    class="w-full relative z-10 py-8 bg-black backdrop-blur-sm border-t border-purple-500/20"
  >
    <div class="max-w-4xl mx-auto text-center px-6 text-sm text-purple-400">
      <div class="space-y-2">
        <p class="font-semibold text-purple-300">
          &copy; 2025 Audio Reprompting Project
        </p>
        <p class="text-purple-400">Universidad de Los Andes</p>
        <p class="text-purple-400">
          Master's Degree in Artificial Intelligence
        </p>
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
