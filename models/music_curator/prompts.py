LECAGY_MUSIC_REPROMPT_PROMPT = """
Compose a prompt to use in a music generation model using cross-modal descriptors, music captions,
and following the rules below:
1. Choose an instrument to lead a melody that aligns with the taste and emotion descriptors.
2. Describe briefly the melody using the next music features: timbre, pitch, and energy, according to the color and human_response descriptors.
3. Purpose music descriptors to indicate articulation, rhythm, and dynamics based on the texture descriptors.
4. Filter the most relevant music captions and temperature descriptors to define one harmony with secondary instruments and music genre.
5. Ensure to use only music descriptors to describe the melody, rhythm, and harmony orchestration.

`Crossmodal descriptors`:
{crossmodal_descriptors}
`Music captions`:
{music_captions}
Follow the rules in steps and order.
Returns only two concise sentences with music descriptors separated by a comma as a result.
"""

MUSIC_REPROMPT_PROMPT_MELODY = """
Compose a prompt to use in a music generation model using cross-modal descriptors, music captions,
and following the rules below:
1. Compose a melody that is crossmodally congruent with the sensation and the emotion descriptors.
2. Describe briefly the melody (pitch, contour, range, tesitura, phrasing), harmony (interval, root, consonance, progression, tonality) and rhythm (beat, pulse, tempo, meter, accent), according to the color and human_response descriptors.
3. Purpose crossmodally congruent music parameters based on somatosensory stimuli.
4. Ensure to orchestrate the melody, harmony, rhythm and genre together to be noticed in the first 30 seconds of the generated music.

`Crossmodal descriptors`:
{crossmodal_descriptors}
`Music captions`:
{music_captions}
Follow the rules in steps and order.
Returns only two concise sentences with music descriptors separated by a comma as a result.
"""

MUSIC_REPROMPT_PROMPT_HARMONY = """
Compose a prompt to use in a music generation model using cross-modal descriptors, music captions,
and following the rules below:
1. Compose a harmony that is crossmodally congruent with the sensation and the emotion descriptors.
2. Describe briefly the melody (pitch, contour, range, tesitura, phrasing), harmony (interval, root, consonance, progression, tonality) and rhythm (beat, pulse, tempo, meter, accent), according to the color and human_response descriptors.
3. Purpose crossmodally congruent music parameters based on somatosensory stimuli.
4. Ensure to orchestrate the melody, harmony, rhythm and genre together to be noticed in the first 30 seconds of the generated music.

`Crossmodal descriptors`:
{crossmodal_descriptors}
`Music captions`:
{music_captions}
Follow the rules in steps and order.
Returns only two concise sentences with music descriptors separated by a comma as a result.
"""

MUSIC_REPROMPT_PROMPT_RHYTHM = """
Compose a prompt to use in a music generation model using cross-modal descriptors, music captions,
and following the rules below:
1. Compose a rhythm that is crossmodally congruent with the sensation and the emotion descriptors.
2. Describe briefly the melody (pitch, contour, range, tesitura, phrasing), harmony (interval, root, consonance, progression, tonality) and rhythm (beat, pulse, tempo, meter, accent), according to the color and human_response descriptors.
3. Purpose crossmodally congruent music parameters based on somatosensory stimuli.
4. Ensure to orchestrate the melody, harmony, rhythm and genre together to be noticed in the first 30 seconds of the generated music.

`Crossmodal descriptors`:
{crossmodal_descriptors}
`Music captions`:
{music_captions}
Follow the rules in steps and order.
Returns only two concise sentences with music descriptors separated by a comma as a result.
"""

MUSIC_REPROMPT_PROMPT_ALL = """
Compose a prompt to use in a music generation model using cross-modal descriptors, music captions,
and following the rules below:
1. Compose a melody, harmony and rhyhtm that are crossmodally congruent with the sensation and the emotion descriptors.
2. Describe briefly the melody (pitch, contour, range, tesitura, phrasing), harmony (interval, root, consonance, progression, tonality) and rhythm (beat, pulse, tempo, meter, accent), according to the color and human_response descriptors.
3. Purpose crossmodally congruent music parameters based on somatosensory stimuli.
4. Ensure to orchestrate the melody, harmony, rhythm and genre together to be noticed in the first 30 seconds of the generated music.

`Crossmodal descriptors`:
{crossmodal_descriptors}
`Music captions`:
{music_captions}
Follow the rules in steps and order.
Returns only two concise sentences with music descriptors separated by a comma as a result.
"""


MCU_PROMPTS = {
    "V1": MUSIC_REPROMPT_PROMPT_MELODY,
    "V2": MUSIC_REPROMPT_PROMPT_HARMONY,
    "V3": MUSIC_REPROMPT_PROMPT_RHYTHM,
    "V4": MUSIC_REPROMPT_PROMPT_ALL,
}
