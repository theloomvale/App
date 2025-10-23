import os
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import gradio as gr

try:
    from huggingface_hub import InferenceClient
except Exception:  # pragma: no cover - optional dependency
    InferenceClient = None

DEFAULT_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
PRO_UNLOCK_CODE = os.getenv("PRO_UNLOCK_CODE", "loomvale-pro").strip().lower()

PLATFORMS = [
    "Instagram Reel",
    "Instagram Carousel",
    "TikTok",
    "Pinterest Pin",
    "YouTube Shorts",
    "LinkedIn",
    "Blog/Post",
    "Newsletter",
]

MOODS = [
    "cozy",
    "dreamy",
    "minimalist",
    "bold",
    "poetic",
    "witty",
    "cinematic",
    "vintage",
    "kawaii",
    "zen",
]

TONES = [
    "Warm storyteller",
    "Data-backed expert",
    "Hype friend",
    "Luxury concierge",
    "Playful mentor",
    "Direct-to-consumer pitch",
]

PERSONAS: Dict[str, Dict[str, List[str]]] = {
    "Wellness Consumers": {
        "pain_points": [
            "Burnout from fast routines",
            "Seeking calm rituals",
            "Overloaded by wellness jargon",
        ],
        "desires": [
            "Simple self-nurture",
            "Proof-backed benefits",
            "Products that feel like a hug",
        ],
        "keywords": ["grounding", "slow rituals", "mind-body reset"],
    },
    "Busy Parents": {
        "pain_points": [
            "Juggling family and self-time",
            "Limited attention spans",
            "Need bite-sized wins",
        ],
        "desires": [
            "Quick wins that feel meaningful",
            "Flexible schedules",
            "Relatable stories",
        ],
        "keywords": ["five-minute reset", "family-first", "real-life demo"],
    },
    "Gen Z Creators": {
        "pain_points": [
            "Scroll fatigue",
            "Skeptical of brand speak",
            "Need authenticity",
        ],
        "desires": [
            "Playful experimentation",
            "Share-worthy hooks",
            "Community-first energy",
        ],
        "keywords": ["duet this", "lofi chaos", "main-character energy"],
    },
    "Premium Shoppers": {
        "pain_points": [
            "Tired of mass-market feel",
            "Need elevated proof",
            "Guarded with trust",
        ],
        "desires": [
            "Luxury cues",
            "White-glove service",
            "Testimonials",
        ],
        "keywords": ["artisan", "limited release", "concierge-level"],
    },
}

TMP_DIR = Path("/tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)


def call_llm(prompt: str, temperature: float = 0.7, max_tokens: int = 512) -> str:
    if not (HF_TOKEN and InferenceClient is not None):
        return ""

    try:
        client = InferenceClient(model=DEFAULT_MODEL, token=HF_TOKEN)
        text = client.text_generation(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            repetition_penalty=1.1,
        )
        return text.strip()
    except Exception:
        return ""


def build_prompt(system_directive: str, user_directive: str) -> str:
    return f"{system_directive}\n\n{user_directive}".strip()


def unlock_pro(code: str) -> Tuple[str, bool]:
    code_normalized = (code or "").strip().lower()
    if code_normalized and code_normalized == PRO_UNLOCK_CODE:
        return (
            "<span style='color:#1B9C85;font-weight:600'>Pro mode unlocked! Enjoy extended outputs, trend deep dives, and export packs.</span>",
            True,
        )
    if not code_normalized:
        return (
            "<span style='color:#E07A5F'>Enter your unlock code to access Pro depth, or grab it via the purchase button.</span>",
            False,
        )
    return (
        "<span style='color:#E07A5F'>That code didn’t match. Double-check your unlock email from Gumroad.</span>",
        False,
    )


def fallback_ideas(seed: str, platform: str, moods: List[str], persona: str, total: int) -> List[str]:
    persona_data = PERSONAS.get(persona, {})
    keywords = ", ".join(persona_data.get("keywords", [])[:2])
    deliverable_hint = {
        "Instagram Reel": "Storyboard the first 3 shots, keep them under 2s each.",
        "Instagram Carousel": "Plan 5 slides with headline, proof, takeaway, CTA, reminder.",
        "TikTok": "Lean on pattern interrupts at second 1.5 and 3.",
        "Pinterest Pin": "Design vertical graphics with layered typography.",
        "YouTube Shorts": "Use kinetic text and ASMR-lite audio cues.",
        "LinkedIn": "Anchor your hook on a metric, close with a reflective question.",
        "Blog/Post": "Break into intro, 3 insights, closing action.",
        "Newsletter": "Segment into letter, resource trio, and micro-challenge.",
    }[platform]

    ideas: List[str] = []
    active_moods = moods or ["cozy"]
    for i in range(total):
        mood = active_moods[i % len(active_moods)]
        headline = f"{mood.title()} take: {seed} for {persona.lower()}"
        flow = f"Frame it around {keywords or 'their daily rhythm'}; finish with {deliverable_hint.lower()}"
        social_proof = "Use a quick stat or testimonial to anchor trust."
        ideas.append(
            f"**{headline}** — Lead with a one-line story, ladder into a transformation, and nod to {persona.lower()} priorities. {flow}. {social_proof}"
        )
    return ideas


def generate_ideas(seed: str, platform: str, moods: List[str], persona: str, total: int, use_llm: bool, is_pro: bool):
    seed = (seed or "").strip()
    if not seed:
        return "Please enter a seed idea to expand.", [], None, None

    capped_total = total if is_pro else min(total, 4)

    ideas: List[str] = []
    if use_llm and HF_TOKEN:
        mood_line = ", ".join(moods) if moods else "cozy"
        system_prompt = (
            "You are a senior social strategist creating multi-platform content blueprints. "
            "Respond in markdown bullet points, weaving in persona-specific insights."
        )
        user_prompt = (
            f"Seed: {seed}. Platform: {platform}. Desired moods: {mood_line}. Persona: {persona}. "
            f"Return {capped_total} distinct ideas. Each idea should include: \n"
            "- Hook angle\n- Story beats\n- Visual or format cue\n- CTA phrased for the persona\n---\n"
            "Separate each idea with a line containing exactly '---'. Keep language punchy and practical."
        )
        raw = call_llm(build_prompt(system_prompt, user_prompt))
        if raw:
            ideas = [item.strip() for item in raw.split("---") if item.strip()]
            ideas = ideas[:capped_total]

    if not ideas:
        ideas = fallback_ideas(seed, platform, moods, persona, capped_total)

    rows = [
        {
            "platform": platform,
            "persona": persona,
            "moods": ", ".join(moods) if moods else "cozy",
            "idea": idea,
        }
        for idea in ideas
    ]

    md_lines = [f"### Idea {idx + 1}\n\n{row['idea']}" for idx, row in enumerate(rows)]
    md = "\n\n---\n\n".join(md_lines)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = TMP_DIR / f"ideas_{ts}.csv"
    json_path = TMP_DIR / f"ideas_{ts}.json"

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["platform", "persona", "moods", "idea"])
        writer.writeheader()
        writer.writerows(rows)

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=2)

    return md, rows, str(csv_path), str(json_path)


def fallback_captions(seed: str, article: str, persona: str, tone: str, platform: str, total: int) -> List[str]:
    persona_data = PERSONAS.get(persona, {})
    headline = seed or (article[:60] + "..." if article else "Your story")
    tone_hint = tone.lower() if tone else "warm storyteller"
    keywords = ", ".join(persona_data.get("keywords", [])[:2])

    captions: List[str] = []
    for idx in range(total):
        hook = f"{headline}: {persona.lower()} deserve better."
        context = (
            f"Pull one proof point or micro-lesson from the article. Translate it into {persona.lower()} language, "
            f"sprinkling phrases like {keywords or 'their daily life'} to ground it."
        )
        cta = "Close with a save/share CTA that promises an immediate win."
        captions.append(
            f"{hook}\n\n{context}\n\nCTA → {cta} (Tone: {tone_hint})."
        )
    return captions


def generate_captions(
    seed: str,
    article: str,
    persona: str,
    tone: str,
    platform: str,
    total: int,
    use_llm: bool,
    is_pro: bool,
):
    article = (article or "").strip()
    seed = (seed or "").strip()
    if not seed and not article:
        return "Paste a seed, article, or notes to generate captions."

    capped_total = total if is_pro else min(total, 3)

    captions: List[str] = []
    if use_llm and HF_TOKEN:
        system_prompt = (
            "You craft persuasive social captions that feel native to each platform and persona. "
            "Return numbered captions with hook, payoff, CTA, and hashtags when relevant."
        )
        user_prompt = (
            f"Persona: {persona}. Tone: {tone}. Platform: {platform}. Number of captions: {capped_total}. "
            f"Seed idea: {seed or 'N/A'}. Source text: {article[:1200]}\n"
            "Each caption should contain: \n"
            "1. Thumb-stopping hook tailored to the persona\n"
            "2. 1-2 sentence story or insight pulled from the source\n"
            "3. CTA geared for conversion\n"
            "4. Optional hashtag line (3-6 hashtags) optimized for the platform\n"
            "Separate each caption with a line containing exactly '---'."
        )
        raw = call_llm(build_prompt(system_prompt, user_prompt), temperature=0.65, max_tokens=700)
        if raw:
            captions = [item.strip() for item in raw.split("---") if item.strip()]
            captions = captions[:capped_total]

    if not captions:
        captions = fallback_captions(seed, article, persona, tone, platform, capped_total)

    return "\n\n---\n\n".join(
        [f"### Caption {idx + 1}\n\n{caption}" for idx, caption in enumerate(captions)]
    )


def persona_deep_dive(persona: str) -> str:
    data = PERSONAS.get(persona)
    if not data:
        return "Select a persona to view insights."

    pain_points = "\n".join(f"- {item}" for item in data["pain_points"])
    desires = "\n".join(f"- {item}" for item in data["desires"])
    keywords = "\n".join(f"- {item}" for item in data["keywords"])

    return (
        f"### {persona} Deep Dive\n\n"
        f"**Pain points**\n{pain_points}\n\n"
        f"**Desired outcomes**\n{desires}\n\n"
        f"**Sticky keywords & phrases**\n{keywords}\n\n"
        "Use these cues to align visuals, ad copy, and landing page messaging."
    )


def build_content_kit(seed: str, platform: str, persona: str, is_pro: bool, use_llm: bool):
    seed = (seed or "").strip()
    if not seed:
        return "Provide a seed to craft the content kit."

    if use_llm and HF_TOKEN:
        depth = "deep-dive" if is_pro else "lite"
        system_prompt = (
            "You are a marketing operator delivering a launch-ready content kit. Return sections in markdown with headings."
        )
        user_prompt = (
            f"Seed: {seed}. Platform: {platform}. Persona: {persona}. Depth level: {depth}."
            "Return the following sections: Hook Headlines (5), Visual Direction, CTA Swaps, Hashtag Stack,"
            " and a 3-post nurture flow (Problem → Proof → Purchase)."
        )
        raw = call_llm(build_prompt(system_prompt, user_prompt), temperature=0.6, max_tokens=750 if is_pro else 500)
        if raw:
            return raw

    persona_data = PERSONAS.get(persona, {})
    hook_templates = [
        f"What if {persona.lower()} could {desire.lower()} starting this week?"
        for desire in persona_data.get("desires", [])
    ] or ["Start with a transformation hook tied to their daily life."]

    visual_direction = (
        f"Lean into {platform.lower()} native cues: mix close-up texture shots with overlay text that echoes '{persona_data.get('keywords', ['their language'])[0]}'"
        if persona_data.get("keywords")
        else f"Use platform-first visuals that mirror the seed '{seed}'."
    )

    cta_swaps = [
        "Send me a DM with 'READY' and I’ll forward the checklist.",
        "Tap shop now and claim your first-order upgrade.",
        "Save this for your reset routine and share with a friend who needs it.",
    ]

    hashtags = [
        "#contentthatconverts",
        "#brandstory",
        "#aestheticstrategy",
        "#consumerinsights",
        "#{platform.replace(' ', '').lower()}tips",
    ]

    nurture_flow = (
        "1. **Problem Post** — spotlight a common frustration and agitate it gently.\n"
        "2. **Proof Post** — share a demo, testimonial, or data-backed transformation.\n"
        "3. **Purchase Post** — unveil the offer with urgency anchored in persona desires."
    )

    return (
        "### Hook Headlines\n"
        + "\n".join(f"- {hook}" for hook in hook_templates)
        + "\n\n### Visual Direction\n"
        + visual_direction
        + "\n\n### CTA Swaps\n"
        + "\n".join(f"- {cta}" for cta in cta_swaps)
        + "\n\n### Hashtag Stack\n"
        + " ".join(hashtags[: (8 if is_pro else 5)])
        + "\n\n### 3-Post Nurture Flow\n"
        + nurture_flow
    )


with gr.Blocks(title="Social Media Post Generator Pro") as demo:
    gr.Markdown(
        """# Social Media Post Generator Pro
Craft end-to-end social campaigns with persona depth, conversion captions, and platform-native guidance.
"""
    )

    with gr.Row():
        gr.Markdown(
            "<div style='display:flex;gap:12px;align-items:center;flex-wrap:wrap'>"
            "<span style='background:#F4F1DE;padding:6px 12px;border-radius:12px;font-weight:600'>Freemium</span>"
            "<span style='background:#1B9C85;color:white;padding:6px 12px;border-radius:12px;font-weight:600'>Purchase Pro</span>"
            "<span>Unlock deeper outputs, persona reports, and export packs.</span>"
            "</div>",
            elem_id="freemium-banner",
        )
        gr.Button("Buy Pro on Gumroad", link="https://loomvale.gumroad.com/l/social-app")

    with gr.Accordion("Have a Pro unlock code?", open=False):
        unlock_code = gr.Textbox(label="Enter unlock code", placeholder="Paste code from your Gumroad receipt")
        unlock_btn = gr.Button("Unlock Pro")
        unlock_status = gr.Markdown()
        pro_state = gr.State(False)
        unlock_btn.click(unlock_pro, inputs=unlock_code, outputs=[unlock_status, pro_state])

    with gr.Tab("Idea Expander"):
        with gr.Row():
            seed = gr.Textbox(label="Brand/Product/Story Seed", lines=2, placeholder="e.g., Adaptogenic cacao launch")
        with gr.Row():
            platform = gr.Dropdown(choices=PLATFORMS, value="Instagram Reel", label="Platform focus")
            persona = gr.Dropdown(choices=list(PERSONAS.keys()), value="Wellness Consumers", label="Audience persona")
            moods = gr.CheckboxGroup(choices=MOODS, value=["cozy", "minimalist"], label="Moods to infuse")
            total = gr.Slider(3, 12, value=6, step=1, label="How many ideas?")
        use_llm = gr.Checkbox(value=True, label="Boost with HF Inference (requires HF_TOKEN)")
        idea_btn = gr.Button("Generate Campaign Ideas ✨")

        idea_md = gr.Markdown()
        idea_df = gr.Dataframe(headers=["platform", "persona", "moods", "idea"], wrap=True)
        idea_csv = gr.File(label="Download ideas CSV")
        idea_json = gr.File(label="Download ideas JSON")

        idea_btn.click(
            generate_ideas,
            inputs=[seed, platform, moods, persona, total, use_llm, pro_state],
            outputs=[idea_md, idea_df, idea_csv, idea_json],
        )

    with gr.Tab("Caption Studio"):
        with gr.Row():
            caption_seed = gr.Textbox(label="Hook or Offer", lines=2, placeholder="e.g., New glow serum with clinical results")
            platform_caption = gr.Dropdown(choices=PLATFORMS, value="Instagram Reel", label="Platform")
        article_notes = gr.Textbox(
            label="Paste article, research, or notes",
            lines=8,
            placeholder="Drop in key paragraphs, research, or a rough draft. We'll extract the strongest beats.",
        )
        with gr.Row():
            caption_persona = gr.Dropdown(choices=list(PERSONAS.keys()), value="Wellness Consumers", label="Persona focus")
            tone = gr.Dropdown(choices=TONES, value="Warm storyteller", label="Voice & tone")
            caption_total = gr.Slider(2, 10, value=4, step=1, label="Number of captions")
        use_llm_caption = gr.Checkbox(value=True, label="Use HF Inference if available")
        caption_btn = gr.Button("Generate Captions & Hashtags ⚡")

        caption_md = gr.Markdown()
        caption_btn.click(
            generate_captions,
            inputs=[
                caption_seed,
                article_notes,
                caption_persona,
                tone,
                platform_caption,
                caption_total,
                use_llm_caption,
                pro_state,
            ],
            outputs=caption_md,
        )

    with gr.Tab("Persona Intelligence"):
        persona_select = gr.Dropdown(choices=list(PERSONAS.keys()), value="Wellness Consumers", label="Select persona")
        persona_md = gr.Markdown(value=persona_deep_dive("Wellness Consumers"))
        persona_select.change(persona_deep_dive, inputs=persona_select, outputs=persona_md)

    with gr.Tab("Content Kit"):
        kit_seed = gr.Textbox(label="Campaign Seed", lines=2, placeholder="e.g., Summer pop-up announcement")
        kit_platform = gr.Dropdown(choices=PLATFORMS, value="Instagram Carousel", label="Platform focus")
        kit_persona = gr.Dropdown(choices=list(PERSONAS.keys()), value="Gen Z Creators", label="Persona")
        use_llm_kit = gr.Checkbox(value=True, label="Use HF Inference if available")
        kit_btn = gr.Button("Build Launch Kit 🚀")
        kit_md = gr.Markdown()
        kit_btn.click(
            build_content_kit,
            inputs=[kit_seed, kit_platform, kit_persona, pro_state, use_llm_kit],
            outputs=kit_md,
        )

    gr.Markdown(
        "---\n"
        "### Export & Workflow Tips\n"
        "- **CSV/JSON exports** capture idea outputs — pass them to Airtable, Notion, or ClickUp.\n"
        "- **Persona intelligence** anchors messaging for consumer-facing launches.\n"
        "- **Caption studio** turns long-form research into scroll-stopping consumer copy.\n"
        "- **Need more?** Tap the Gumroad button to unlock automations, extended templates, and live updates."
    )

if __name__ == "__main__":
    demo.launch()
