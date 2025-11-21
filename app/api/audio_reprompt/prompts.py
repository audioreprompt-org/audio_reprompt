MUSIC_REPROMPT_PROMPT_V3 = """
Compose a prompt to use in a music generation model using cross-modal descriptors, music captions,
 and following the rules below:
1. Choose an instrument to lead a melody that aligns with the taste and emotion descriptors.
2. Describe briefly the melody, mapping the color and human_response descriptors to the next music features: (timbre, pitch, and energy).
3. Purpose music descriptors to indicate articulation, rhythm, and dynamics based on the texture and emotion descriptors.
4. Using the provided music captions and the temperature descriptors, filter the most relevant to define one harmony with secondary instruments to integrate with the melody.
5. Ensure to orchestrate the melody and harmony together to be noticed in the first 30 seconds of the generated music. 

`Crossmodal descriptors`:
{crossmodal_descriptors}
`Music captions`:
{music_captions}
Follow the rules in steps and order.
Returns only a paragraph with two concise complex sentences as a result, and ensures to cover the rules provided.
"""

# We can add more if needed
MCU_PROMPTS = {
    "V3": MUSIC_REPROMPT_PROMPT_V3,
}
