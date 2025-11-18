<script setup>
import { ref } from "vue";
import PromptScreen from "./components/PromptScreen.vue";
import PlaybackScreen from "./components/PlaybackScreen.vue";
import { useApi } from "./composables/useApi.js";

const currentScreen = ref("prompt"); // 'prompt' | 'playback'
const { data, loading, error, generateMusic } = useApi();

const handleGenerate = async (prompt) => {
  await generateMusic(prompt);

  if (error.value) {
    console.error("Error generating music:", error.value);
    currentScreen.value = "prompt";
  } else {
    // CORRECCIÓN 2: Asegurar que el cambio a 'playback' solo ocurre después de la carga
    currentScreen.value = "playback";
  }
};

const handleReset = () => {
  // 1. Resetear datos y errores
  data.value = null;
  error.value = null;
  currentScreen.value = "prompt";
};
</script>

<template>
  <div id="app" class="font-sans antialiased">
    <PromptScreen
      v-if="currentScreen === 'prompt'"
      @generate="handleGenerate"
      :is-loading="loading"
    />

    <PlaybackScreen
      v-else-if="currentScreen === 'playback'"
      :music-data="data"
      @reset-and-go-back="handleReset"
    />
  </div>
</template>
