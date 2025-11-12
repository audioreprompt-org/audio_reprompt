<script setup>
import { ref, onUnmounted } from "vue";

// Define event emitter for parent communication (App.vue listens to "generate").
const emit = defineEmits(["generate"]);
// Default example text for the prompt input.
const DEFAULT_PROMPT =
  "E.g. I'm at the food court and I'm gonna eat a bowl of ramen. The space is noisy and the neon signs are bright.";

// Validation thresholds.
const MIN_LENGTH = 20; // Minimum number of characters.
const MAX_LENGTH = 500; // Maximum number of characters.
const MIN_WORDS = 3; // Minimum number of words (for descriptive prompts).

// Validation toggles.
const isCharacterLengthEnabled = ref(true); // Enables/disables strict length validation.
const isSanitizationEnabled = ref(true); // Enables/disables security validation.
const isLanguageCheckEnabled = ref(true); // Enables/disables English-only check.

// Reactive state for prompt and validation feedback.
const promptText = ref(DEFAULT_PROMPT);
const showError = ref(false);
const errorMessage = ref(
  "Please, enter a prompt to generate a piece of music."
);

// Regex patterns to detect XSS, HTML/script injection, SQL commands, or unsafe input.
const DANGER_PATTERNS = [
  /<[a-zA-Z\/].*?>/, // HTML or script tags.
  /SELECT\s/i, // SQL injection keyword.
  /DROP\s/i, // SQL injection keyword.
  /DELETE\s/i, // SQL injection keyword.
  /('|")/, // Quotes potentially closing SQL strings.
  /;/, // Semicolons chaining shell/SQL commands.
];

// Rotating loading messages to simulate creative generation progress.
const messages = [
  "Capturing the mood of your moment…",
  "Listening to your words, feeling the atmosphere…",
  "Translating your scene into melodies and textures…",
  "Shaping harmonies that match your story…",
  "Mixing sounds that taste like your emotions…",
  "Balancing rhythm and ambience…",
  "Adding a touch of warmth and depth to the composition…",
  "Letting the music breathe your moment…",
  "Almost finished, I'm working on the final notes……",
  "Your soundscape is ready to unfold…",
];

const currentMessageIndex = ref(0);
const currentMessage = ref(messages[0]);
const rotationInterval = 3500;
const loadingKey = ref(0);
let messageTimer = null;
let initialTimeout = null;

const advanceMessage = () => {
  /**
   * Advances to the next message in the loading cycle.
   */
  if (currentMessageIndex.value < messages.length - 1) {
    currentMessageIndex.value++;
    currentMessage.value = messages[currentMessageIndex.value];
  }
};

const stopMessageRotation = () => {
  /**
   * Stops and clears all active timers for message rotation.
   */
  if (messageTimer) {
    clearInterval(messageTimer);
    messageTimer = null;
  }
  if (initialTimeout) {
    clearTimeout(initialTimeout);
    initialTimeout = null;
  }
  // Reset to the first message.
  currentMessageIndex.value = 0;
  currentMessage.value = messages[0];
};

const startMessageRotation = () => {
  /**
   * Starts cycling through the loading messages during generation.
   */
  loadingKey.value++; // Force re-render if loading restarts.
  stopMessageRotation();

  // Trigger first message quickly after component update.
  initialTimeout = setTimeout(() => {
    advanceMessage(); // Muestra el mensaje[1]

    messageTimer = setInterval(advanceMessage, rotationInterval);
  }, 10); // 10ms.
};

// Ensure timers are cleared when component unmounts.
onUnmounted(() => {
  stopMessageRotation();
});

// Receives the `isLoading` prop from the parent (App.vue).
const props = defineProps({
  isLoading: {
    type: Boolean,
    default: false,
  },
});

const checkLanguage = (prompt) => {
  /**
   * Checks if the prompt is mostly English (simple Latin characters only).
   * This is a lightweight client-side heuristic.
   * @param {string} prompt - User input to validate.
   * @returns {boolean} - True if the text is acceptable, false otherwise.
   */
  if (!isLanguageCheckEnabled.value) {
    return true;
  }

  const latinRegex = /^[a-z0-9\s.,?!'":;@#-]+$/i;

  if (!latinRegex.test(prompt)) {
    errorMessage.value =
      "Language Warning: Please ensure your prompt is in English (Latin alphabet only) for optimal results.";
    return false;
  }
  return true;
};

const sanitizePrompt = (prompt) => {
  /**
   * Detects unsafe input such as HTML, SQL, or script injections.
   * @param {string} prompt - User input to sanitize.
   * @returns {boolean} - True if the input is clean, false if dangerous.
   */

  if (!isSanitizationEnabled.value) {
    return true;
  }

  for (const pattern of DANGER_PATTERNS) {
    if (pattern.test(prompt)) {
      errorMessage.value =
        "Security warning: The prompt contains inappropriate or dangerous characters (e.g., HTML tags or script commands). Please use only descriptive text.";
      return false;
    }
  }
  return true;
};

const validatePrompt = (prompt) => {
  /**
   * Validates the prompt’s length and descriptiveness.
   * @param {string} prompt - User input after sanitization.
   * @returns {boolean} - True if valid, false otherwise.
   */
  if (!isCharacterLengthEnabled.value) {
    return true; // Validación desactivada.
  }

  const charCount = prompt.length;
  const wordCount = prompt
    .split(/\s+/)
    .filter((word) => word.length > 0).length;

  // Check for overly long prompts.
  if (charCount > MAX_LENGTH) {
    errorMessage.value = `Prompt is too long (${charCount} characters). Maximum allowed is ${MAX_LENGTH} characters.`;
    return false;
  }

  // Check for minimal descriptive input.
  const passesMinimum = charCount >= MIN_LENGTH || wordCount >= MIN_WORDS;

  if (!passesMinimum) {
    errorMessage.value = `Prompt is too short. Please provide at least ${MIN_LENGTH} characters or ${MIN_WORDS} descriptive words.`;
    return false;
  }

  return true;
};

const submitPrompt = () => {
  /**
   * Main submit handler:
   * Validates the user prompt and emits the "generate" event if valid.
   */
  const userPrompt = promptText.value.trim();
  const isDefaultPrompt = userPrompt === DEFAULT_PROMPT;
  const cleanedPrompt = isDefaultPrompt ? "" : userPrompt;

  // 1. Basic empty check.
  if (cleanedPrompt.length === 0) {
    errorMessage.value = "Please enter a prompt to generate music.";
    showError.value = true;
    return;
  }

  // 2. Security validation.
  if (!sanitizePrompt(userPrompt)) {
    showError.value = true;
    return;
  }

  // 3. Language validation.
  if (!checkLanguage(userPrompt)) {
    showError.value = true;
    return;
  }

  // 4. Length/content validation.
  if (!validatePrompt(userPrompt)) {
    showError.value = true;
    return;
  }

  // If all checks pass, trigger generation.
  showError.value = false;
  if (!props.isLoading) {
    startMessageRotation();
  }
  emit("generate", userPrompt);
};

const clearPrompt = () => {
  /**
   * Clears placeholder text when user focuses the input field.
   */
  if (promptText.value === DEFAULT_PROMPT) {
    promptText.value = "";
  }
  showError.value = false; // Oculta el error si el usuario intenta escribir.
  errorMessage.value = "Please enter a prompt to generate a piece of music."; // Resetear mensaje.
};

const resetPrompt = () => {
  /**
   * Restores default prompt if the field is left empty after blur.
   */
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

    <div class="relative z-10 max-w-4xl w-full text-center mt-10 space-y-16">
      <h1
        class="text-4xl font-extrabold tracking-tight mb-20 text-white [text-shadow:_0_0_4px_rgba(168,85,247,0.35),_0_0_8px_rgba(168,85,247,0.25)]"
      >
        Imagine a piece of music to accompany your meal
      </h1>

      <p class="text-xl text-white px-6">
        Where are you, what are you going to eat, and who are you sharing this
        moment with?
      </p>

      <!--
      <div
        class="text-sm text-white/50 cursor-pointer flex justify-center space-x-6"
      >
        <span @click="isLanguageCheckEnabled = !isLanguageCheckEnabled">
          Length Validation: {{ isLanguageCheckEnabled ? "ON ✅" : "OFF ❌" }}
        </span>
        <span @click="isSanitizationEnabled = !isSanitizationEnabled">
          Security Filter: {{ isSanitizationEnabled ? "ON 🔒" : "OFF 🔓" }}
        </span>
        <span @click="isLanguageCheckEnabled = !isLanguageCheckEnabled">
          Language Check: {{ isLanguageCheckEnabled ? "ON 🌎" : "OFF 🌍" }}
        </span>
      </div>
      -->

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
        {{ errorMessage }}
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
        :key="loadingKey"
        class="flex flex-col items-center space-y-8"
      >
        <div
          class="flex justify-center items-center space-x-6 mt-2 text-7xl font-bold [filter:drop-shadow(0_0_10px_rgb(167,139,250))_drop-shadow(0_0_10px_rgb(79,70,229))]"
        >
          <span
            class="text-main-blue animate-pulse [text-shadow:_0_0_1px_#000000]"
            style="animation-delay: 0s"
          >
            &#9839;
          </span>
          <span
            class="text-main-purple animate-pulse [text-shadow:_0_0_1px_#000000]"
            style="animation-delay: 0.15s"
          >
            &#9836;
          </span>
          <span
            class="text-light-purple animate-pulse [text-shadow:_0_0_1px_#000000]"
            style="animation-delay: 0.3s"
          >
            &#9835;
          </span>
          <span
            class="text-main-blue animate-pulse [text-shadow:_0_0_1px_#000000]"
            style="animation-delay: 0.45s"
          >
            &#9837;
          </span>
          <span
            class="text-main-purple animate-pulse [text-shadow:_0_0_1px_#000000]"
            style="animation-delay: 0.6s"
          >
            &#119070;
          </span>
        </div>
        <p class="text-lg text-purple-300 font-semibold">
          {{ currentMessage }}
        </p>
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
