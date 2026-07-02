"""Image Generator (seed skill).

A main-agent skill that unlocks image generation. Activating it surfaces the
``chat_generate_image`` tool (``section="skills"``, ``audience="main"``) plus the
instructions for using it. The tool additionally requires a configured image
model — when none resolves it is withheld even while the skill is active (see
core/preferences.py). Sub-agents do not get this tool.
"""

IMAGE_GENERATOR = {
    "slug": "image_generator",
    "name": "Image Generator",
    "emoji": "🎨",
    "description": (
        "Generate or edit images from a text prompt. Activate this when the user "
        "wants to create a picture or illustration, or to restyle/edit an existing image. "
        "**Note:** This skill has tools that enable calling image models for generation and editing."
    ),
    "instructions": """\
# Image Generator

Generate or edit images.

## Creation:

Use `chat_generate_image` with a prompt describing the image you want to create.

## Editing:

To EDIT or restyle an existing image (one you just generated, or a data-room image
you've viewed), pass its `[[image:<uuid>|]]` token in `input_images` together with a
prompt describing the change. The tool saves the image and returns a token — paste
that token into your reply to show it to the user.

As a general rule, generate ONCE and present the result; don't regenerate or iterate
on your own unless the user asks (or the image clearly missed what was requested).

## House style:

Unless asked for a particular style, use this suffix as part of the prompt. It's your "house style":

>Editorial documentary photograph, real research/innovation environment, warm natural window light, muted low-contrast color grade with a subtle deep-forest-green tint in the shadows, fine film grain, shallow-to-medium depth of field, calm and precise mood, generous negative space, no text, no logos, no visible faces of recognizable people, not glossy stock, no handshakes-on-white, no tilted angles, photorealistic, 35mm.

## Communication

After generation, always relay the exact prompt used back to the user, together with the image produced.

The part after the `|` \
is an **optional caption** that becomes the image's alt text — leave it empty, or write your own \
short caption between the `|` and the `]]` (e.g. `[[image:<uuid>|Figure 1: aerial view of the test \
facility]]`). You can also set the image's display width by appending `|NN%` after the caption, \
where NN is 10–100 (percent of the available width) — e.g. `[[image:<uuid>|Figure 1: site map|60%]]`. \
Omit it to show the image at full width; use a smaller width for things like logos or portraits that \
look oversized full-bleed. Two sized tokens written right next to each other (no blank line between) \
render side by side on one line — handy for a pair of images; put a blank line between them to stack \
them vertically instead. The image renders inline in the preview and the chat, and is baked into any \
.docx or .pdf export.

Do NOT write markdown image syntax such as `![caption](file.png)` — it does not render and is stripped out. Never invent a token uuid — only use one a tool gave you.

""",
    "tool_names": ["chat_generate_image"],
}
