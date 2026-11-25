"""StudioMode Production Bible (project.json) compiler and stateful continuity resolver for Nairobiwood."""

from __future__ import annotations

import copy
import json
import re
import shutil
from pathlib import Path
from typing import Any

from comfy_factory_mcp.config import FactorySettings
from comfy_factory_mcp.models import CharacterSpec, KeyframeState, ShotManifest, StoryManifest, utc_now_iso


CANONICAL_BIBLE_TEMPLATE: dict[str, Any] = {
    "schema_version": "1.1.0",
    "project": {
        "id": "",
        "title": "",
        "logline": "",
        "source_story_path": "",
        "workflow_stage": "storyboarding",
        "execution_backend": "nairobiwood",
        "technical_specs": {
            "aspect_ratio": "16:9",
            "resolution": "1080p",
            "fps": 24,
            "target_duration_sec": None,
        },
        "higgsfield_config": {
            "mcp_url": "https://mcp.higgsfield.ai/mcp",
            "rest_api_base": "https://api.higgsfield.ai",
            "draft_quality": "turbo",
            "final_quality": "1080p",
            "model_routing": {
                "character_portraits_human": "soul_v2",
                "creatures_props_locations": "flux_2",
                "fast_storyboard_drafts": "nano_banana_pro",
                "cinematic_video_default": "cinema_studio",
                "native_dialogue_video": "veo_3_1",
                "multi_subject_ref_video": "seedance_2_5",
            },
        },
    },
    "alignment_board": {
        "story_and_photography_aligned": True,
        "scripter_open_questions": [],
        "dop_observations": [],
    },
    "style_guide": {
        "locked": True,
        "visual_medium": "Grounded 35mm anamorphic theatrical cinema, tactile skin and fabric texture, Kodak Vision3 500T grain",
        "genre_and_tone": "Cinematic narrative drama",
        "world_and_era": "Contemporary",
        "lighting_philosophy": "Motivated practical and natural lighting with rich shadow contrast",
        "color_script": {
            "mode": "dynamic_per_scene",
            "global_palette": "Natural filmic contrast",
            "act_or_scene_progression": "",
        },
        "cinema_studio_optics": {
            "camera_body": "ARRI Alexa 35",
            "lens_package": "Atlas Orion Anamorphic primes",
            "focal_lengths_mm": [24, 35, 50, 85],
            "default_aperture": "f/2.0",
            "texture_and_grain": "Fine 35mm Kodak Vision3 500T film grain",
            "higgsfield_style_id": None,
        },
        "style_references": {
            "user_reference_photos": [],
            "generated_lookbook_urls": [],
            "approved_reference_urls": [],
        },
    },
    "assets": {
        "characters": [],
        "locations": [],
        "props": [],
    },
    "scenes": [],
}


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "unnamed"


def guess_entity_type(name: str, role: str, appearance: str) -> str:
    combined = f"{name} {role} {appearance}".lower()
    if name.strip().lower().endswith("s") and any(
        w in combined for w in ("villagers", "crowd", "people", "guards", "citizens", "elders")
    ):
        return "crowd"
    creature_keywords = (
        "frog", "chameleon", "lion", "bird", "snake", "spider", "tortoise",
        "hare", "dog", "cat", "wolf", "dragon", "animal", "creature", "beast",
    )
    if any(re.search(rf"\b{kw}\b", combined) for kw in creature_keywords):
        return "creature"
    return "human"


def make_character_entry(
    char_id: str,
    name: str,
    source_id: Any = None,
    gender: str = "",
    role: str = "",
    appearance: str = "",
    default_wardrobe: str = "",
    default_prop_ids: list[str] | None = None,
    personality: str = "",
    entity_type: str = "human",
) -> dict[str, Any]:
    use_soul = entity_type == "human"
    return {
        "id": char_id,
        "source_id": source_id,
        "name": name,
        "entity_type": entity_type,
        "gender": gender,
        "role": role,
        "appearance": appearance,
        "default_wardrobe": default_wardrobe or appearance,
        "default_prop_ids": list(default_prop_ids or []),
        "personality": personality,
        "scale_and_blocking_notes": "",
        "voice_profile": "",
        "casting": {
            "source_mode": "unassigned",
            "local_reference_dir": f"references/characters/{char_id}",
            "master_portrait_path": None,
            "user_reference_photos": [],
            "generated_turnaround_urls": [],
            "approved_reference_urls": [],
            "higgsfield_character": {
                "use_soul_id": use_soul,
                "character_id": None,
                "custom_reference_strength": 0.85,
                "status": "unregistered" if use_soul else "image_refs_only",
            },
            "nairobiwood": {
                "mode": "kontext_portrait" if use_soul else "description_only",
                "lora_name": None,
                "trigger_word": "",
                "lora_strength": 0.85,
            },
        },
    }


def parse_raw_story_into_bible(raw: dict[str, Any], project_slug: str, title: str | None = None) -> dict[str, Any]:
    """Compile a raw Story JSON (Story.Character / Story.Scenes) into a canonical StudioMode project.json."""
    project = copy.deepcopy(CANONICAL_BIBLE_TEMPLATE)
    story = raw.get("Story") or raw.get("story") or raw
    story_title = str(story.get("Title") or story.get("title") or title or project_slug).strip()
    logline = str(story.get("Logline") or story.get("logline") or "").strip()

    project["project"]["id"] = project_slug
    project["project"]["title"] = story_title
    project["project"]["logline"] = logline

    raw_chars = story.get("Character") or story.get("Characters") or story.get("characters") or []
    raw_props = story.get("Props") or story.get("props") or []
    raw_scenes = story.get("Scenes") or story.get("scenes") or []

    char_by_name: dict[str, str] = {}
    characters: list[dict[str, Any]] = []

    for idx, c in enumerate(raw_chars, start=1):
        name = str(c.get("Name") or c.get("name") or f"Character {idx}").strip()
        cid = c.get("id") or f"char_{slugify(name)}"
        role = str(c.get("Rol/Description") or c.get("Role") or c.get("role") or c.get("description") or "").strip()
        appearance = str(c.get("Appearance") or c.get("appearance") or "").strip()
        wardrobe = str(c.get("Wardrobe") or c.get("default_wardrobe") or appearance).strip()
        def_props = list(c.get("default_prop_ids") or c.get("Props") or [])
        personality = str(c.get("Personality") or c.get("personality") or "").strip()
        gender = str(c.get("Gender") or c.get("gender") or "").strip()
        etype = guess_entity_type(name, role, appearance)

        entry = make_character_entry(
            char_id=cid,
            name=name,
            source_id=c.get("Id", c.get("id", idx)),
            gender=gender,
            role=role,
            appearance=appearance,
            default_wardrobe=wardrobe,
            default_prop_ids=def_props,
            personality=personality,
            entity_type=etype,
        )
        characters.append(entry)
        char_by_name[name.lower()] = cid

    props: list[dict[str, Any]] = []
    for p_idx, p in enumerate(raw_props, start=1):
        pname = str(p.get("Name") or p.get("name") or f"Prop {p_idx}").strip()
        pid = str(p.get("id") or f"prop_{slugify(pname)}")
        props.append(
            {
                "id": pid,
                "name": pname,
                "description": str(p.get("Description") or p.get("description") or pname).strip(),
                "lighting_default": str(p.get("lighting_default") or ""),
                "scene_ids": list(p.get("scene_ids") or []),
                "references": {
                    "source_mode": "unassigned",
                    "local_reference_dir": f"references/props/{pid}",
                    "user_reference_photos": [],
                    "generated_plate_urls": [],
                    "approved_reference_urls": [],
                },
            }
        )

    locations: list[dict[str, Any]] = []
    loc_by_desc: dict[str, str] = {}
    scenes: list[dict[str, Any]] = []

    for s_idx, s in enumerate(raw_scenes, start=1):
        scene_id = str(s.get("id") or f"scene_{s_idx:02d}")
        s_name = str(s.get("Scene_Name") or s.get("name") or f"Scene {s_idx}").strip()
        s_plot = str(s.get("Scene_plot") or s.get("plot_summary") or s.get("plot") or "").strip()
        s_bg = str(s.get("Scene_background") or s.get("background") or "").strip()

        bg_key = s_bg.lower() if s_bg else f"location_{s_idx}"
        if bg_key not in loc_by_desc:
            loc_id = f"loc_{slugify(s_name)}"
            existing_loc_ids = {loc["id"] for loc in locations}
            if loc_id in existing_loc_ids:
                loc_id = f"{loc_id}_{s_idx:02d}"
            loc_by_desc[bg_key] = loc_id
            locations.append(
                {
                    "id": loc_id,
                    "name": f"{s_name} Setting",
                    "description": s_bg,
                    "lighting_default": "",
                    "scene_ids": [scene_id],
                    "references": {
                        "source_mode": "unassigned",
                        "local_reference_dir": f"references/locations/{loc_id}",
                        "user_reference_photos": [],
                        "generated_plate_urls": [],
                        "approved_reference_urls": [],
                    },
                }
            )
        else:
            loc_id = loc_by_desc[bg_key]
            for loc in locations:
                if loc["id"] == loc_id and scene_id not in loc["scene_ids"]:
                    loc["scene_ids"].append(scene_id)

        scene_char_ids: list[str] = []
        mentioned_names = list(s.get("Scenes_Characters") or s.get("character_ids") or s.get("characters") or [])
        raw_dialogues = s.get("Scene_dialogues") or s.get("dialogues") or []
        for d in raw_dialogues:
            speaker = str(d.get("Character_speaking") or d.get("character") or "").strip()
            if speaker and speaker not in mentioned_names:
                mentioned_names.append(speaker)

        for m_name in mentioned_names:
            clean_name = str(m_name).strip()
            if not clean_name:
                continue
            key = clean_name.lower()
            if clean_name.startswith("char_") and any(c["id"] == clean_name for c in characters):
                cid = clean_name
            elif key not in char_by_name:
                cid = f"char_{slugify(clean_name)}"
                etype = guess_entity_type(clean_name, "Implicit scene character", "")
                characters.append(
                    make_character_entry(
                        char_id=cid,
                        name=clean_name,
                        source_id=None,
                        role="Implicit character extracted from scene breakdown",
                        entity_type=etype,
                    )
                )
                char_by_name[key] = cid
            else:
                cid = char_by_name[key]
            if cid not in scene_char_ids:
                scene_char_ids.append(cid)

        beats: list[dict[str, Any]] = list(s.get("beats") or [])
        if not beats:
            for b_idx, d in enumerate(raw_dialogues):
                speaker = str(d.get("Character_speaking") or d.get("character") or "").strip()
                speaker_id = char_by_name.get(speaker.lower()) if speaker else (scene_char_ids[0] if scene_char_ids else None)
                text = str(d.get("Text") or d.get("text") or "").strip()
                tone = str(d.get("Tone") or d.get("tone") or "").strip()
                try:
                    order = int(d.get("Sequential", b_idx))
                except (TypeError, ValueError):
                    order = b_idx
                beats.append(
                    {
                        "id": f"s{s_idx:02d}_b{b_idx:02d}",
                        "order": order,
                        "type": "dialogue",
                        "character_id": speaker_id,
                        "tone": tone,
                        "blocking_action": s_plot,
                        "spoken_text": text,
                    }
                )
        if not beats:
            beats.append(
                {
                    "id": f"s{s_idx:02d}_b00",
                    "order": 0,
                    "type": "action",
                    "character_id": scene_char_ids[0] if scene_char_ids else None,
                    "tone": "cinematic",
                    "blocking_action": s_plot or s_name,
                    "spoken_text": None,
                }
            )

        raw_char_states = s.get("Character_states") or s.get("character_states") or []
        state_by_cid: dict[str, dict[str, Any]] = {}
        for cs in raw_char_states:
            raw_cname = str(cs.get("character_id") or cs.get("Character") or cs.get("name") or "").strip()
            resolved_cid = raw_cname if raw_cname.startswith("char_") else char_by_name.get(raw_cname.lower())
            if resolved_cid:
                state_by_cid[resolved_cid] = cs

        # Also support top-level scene Wardrobe / Props overrides if provided on the scene object
        scene_wardrobe_map = s.get("Wardrobe") or s.get("wardrobe_overrides") or {}
        scene_props_raw = s.get("Props") or s.get("prop_ids") or []
        scene_prop_ids: list[str] = []
        for sp in scene_props_raw:
            sp_str = str(sp).strip()
            if not sp_str:
                continue
            if sp_str.startswith("prop_"):
                pid = sp_str
            else:
                pid = f"prop_{slugify(sp_str)}"
            if not any(existing_p["id"] == pid for existing_p in props):
                props.append(
                    {
                        "id": pid,
                        "name": sp_str,
                        "description": sp_str,
                        "lighting_default": "",
                        "scene_ids": [scene_id],
                        "references": {
                            "source_mode": "unassigned",
                            "local_reference_dir": f"references/props/{pid}",
                            "user_reference_photos": [],
                            "generated_plate_urls": [],
                            "approved_reference_urls": [],
                        },
                    }
                )
            scene_prop_ids.append(pid)

        character_states: list[dict[str, Any]] = []
        for cid in scene_char_ids:
            cs_raw = state_by_cid.get(cid, {})
            cname = next((ch["name"] for ch in characters if ch["id"] == cid), cid)
            w_override = (
                cs_raw.get("wardrobe")
                or cs_raw.get("Wardrobe")
                or scene_wardrobe_map.get(cid)
                or scene_wardrobe_map.get(cname)
                or None
            )
            p_override_raw = cs_raw.get("prop_ids") if "prop_ids" in cs_raw else cs_raw.get("Props")
            p_override: list[str] | None = None
            if isinstance(p_override_raw, list):
                p_override = []
                for pr in p_override_raw:
                    pr_str = str(pr).strip()
                    pid = pr_str if pr_str.startswith("prop_") else f"prop_{slugify(pr_str)}"
                    if not any(ep["id"] == pid for ep in props):
                        props.append(
                            {
                                "id": pid,
                                "name": pr_str,
                                "description": pr_str,
                                "lighting_default": "",
                                "scene_ids": [scene_id],
                                "references": {
                                    "source_mode": "unassigned",
                                    "local_reference_dir": f"references/props/{pid}",
                                    "user_reference_photos": [],
                                    "generated_plate_urls": [],
                                    "approved_reference_urls": [],
                                },
                            }
                        )
                    p_override.append(pid)

            character_states.append(
                {
                    "character_id": cid,
                    "wardrobe": w_override,
                    "prop_ids": p_override,
                    "continuity_mode": "override" if (w_override or p_override) else "inherit_prior_scene",
                    "resolved_wardrobe": "",
                    "resolved_prop_ids": [],
                }
            )

        scenes.append(
            {
                "id": scene_id,
                "source_id": s.get("Id", s.get("id", s_idx)),
                "order": s_idx,
                "name": s_name,
                "plot_summary": s_plot,
                "location_id": loc_id,
                "character_ids": scene_char_ids,
                "prop_ids": scene_prop_ids,
                "character_states": character_states,
                "dop_direction": s.get("dop_direction")
                or {
                    "time_of_day": "Cinematic natural / practical lighting",
                    "lighting_setup": s_bg or "Motivated key light with deep filmic contrast",
                    "color_override": "",
                    "lens_strategy": "35mm & 50mm anamorphic primes",
                    "camera_energy": "Controlled dolly and tracking movement",
                },
                "beats": beats,
                "shots": list(s.get("shots") or []),
            }
        )

    project["assets"]["characters"] = characters
    project["assets"]["locations"] = locations
    project["assets"]["props"] = props
    project["scenes"] = scenes

    resolve_scene_continuity(project)
    synthesize_missing_shots(project)
    resolve_scene_continuity(project)
    return project


def resolve_scene_continuity(project: dict[str, Any]) -> dict[str, Any]:
    """Propagate character wardrobe and active props from Scene N-1 to Scene N when Scene N has no override."""
    project.setdefault("schema_version", "1.1.0")
    chars_by_id: dict[str, dict[str, Any]] = {}
    for ch in project.get("assets", {}).get("characters", []):
        cid = ch["id"]
        ch.setdefault("default_wardrobe", ch.get("appearance", ""))
        ch.setdefault("default_prop_ids", [])
        casting = ch.setdefault("casting", {})
        casting.setdefault("master_portrait_path", None)
        casting.setdefault(
            "nairobiwood",
            {
                "mode": "kontext_portrait" if ch.get("entity_type", "human") == "human" else "description_only",
                "lora_name": None,
                "trigger_word": "",
                "lora_strength": 0.85,
            },
        )
        chars_by_id[cid] = ch

    props_by_id: dict[str, dict[str, Any]] = {
        p["id"]: p for p in project.get("assets", {}).get("props", [])
    }

    active_wardrobe: dict[str, str] = {
        cid: str(ch.get("default_wardrobe") or ch.get("appearance") or "").strip()
        for cid, ch in chars_by_id.items()
    }
    active_props: dict[str, list[str]] = {
        cid: list(ch.get("default_prop_ids") or [])
        for cid, ch in chars_by_id.items()
    }
    last_scene_props: list[str] = []

    scenes = sorted(project.get("scenes", []), key=lambda sc: sc.get("order", 0))
    for scene in scenes:
        explicit_scene_props = scene.get("prop_ids")
        if explicit_scene_props:
            last_scene_props = list(explicit_scene_props)
        elif last_scene_props and not explicit_scene_props:
            scene["prop_ids"] = list(last_scene_props)

        existing_states = {
            cs.get("character_id"): cs
            for cs in scene.get("character_states", [])
            if isinstance(cs, dict) and cs.get("character_id")
        }
        updated_states: list[dict[str, Any]] = []
        for cid in scene.get("character_ids", []):
            cs = existing_states.get(
                cid,
                {
                    "character_id": cid,
                    "wardrobe": None,
                    "prop_ids": None,
                    "continuity_mode": "inherit_prior_scene",
                },
            )
            w_override = (cs.get("wardrobe") or "").strip() if isinstance(cs.get("wardrobe"), str) else None
            p_override = cs.get("prop_ids") if isinstance(cs.get("prop_ids"), list) else None

            if w_override:
                active_wardrobe[cid] = w_override
                w_mode = "override"
            else:
                w_mode = "inherit_prior_scene"

            if p_override is not None and len(p_override) > 0:
                active_props[cid] = list(p_override)
                p_mode = "override"
            elif not active_props.get(cid) and scene.get("prop_ids"):
                active_props[cid] = list(scene["prop_ids"])
                p_mode = "inherit_scene_props"
            else:
                p_mode = "inherit_prior_scene"

            cs["wardrobe"] = w_override
            cs["prop_ids"] = p_override
            cs["continuity_mode"] = "override" if ("override" in (w_mode, p_mode)) else "inherit_prior_scene"
            cs["resolved_wardrobe"] = active_wardrobe.get(cid, "")
            cs["resolved_prop_ids"] = list(active_props.get(cid, []))
            updated_states.append(cs)

        scene["character_states"] = updated_states
        state_lookup = {cs["character_id"]: cs for cs in updated_states}

        for shot in scene.get("shots", []):
            bindings = shot.setdefault("bindings", {})
            bound_cids = bindings.get("character_ids") or scene.get("character_ids", [])
            master_portraits: dict[str, str] = {}
            inh_wardrobe: dict[str, str] = {}
            inh_props: dict[str, list[str]] = {}
            prop_descs: list[str] = []
            seen_pids: set[str] = set()

            for cid in bound_cids:
                ch = chars_by_id.get(cid, {})
                mp = ch.get("casting", {}).get("master_portrait_path")
                if mp:
                    master_portraits[cid] = mp
                c_state = state_lookup.get(cid, {})
                inh_wardrobe[cid] = c_state.get("resolved_wardrobe") or active_wardrobe.get(cid, "")
                pids = c_state.get("resolved_prop_ids") or active_props.get(cid, [])
                inh_props[cid] = list(pids)
                for pid in pids:
                    if pid not in seen_pids and pid in props_by_id:
                        seen_pids.add(pid)
                        p_obj = props_by_id[pid]
                        prop_descs.append(f"{p_obj.get('name', pid)}: {p_obj.get('description', '')}".strip(": "))

            bindings["continuity"] = {
                "master_portrait_paths": master_portraits,
                "inherited_wardrobe": inh_wardrobe,
                "inherited_prop_ids": inh_props,
                "inherited_prop_descriptions": prop_descs,
            }

    return project


def synthesize_missing_shots(project: dict[str, Any]) -> None:
    """When a raw story is imported without pre-broken shots[], compile 6-layer start_frame & video motion prompts."""
    style = project.get("style_guide", {})
    medium = style.get("visual_medium") or "Grounded 35mm anamorphic cinema"
    optics = style.get("cinema_studio_optics", {})
    cam_body = optics.get("camera_body") or "ARRI Alexa 35"
    lens_pkg = optics.get("lens_package") or "Atlas Orion Anamorphic lens"
    aperture = optics.get("default_aperture") or "f/2.0"

    chars_by_id = {c["id"]: c for c in project.get("assets", {}).get("characters", [])}
    locs_by_id = {l["id"]: l for l in project.get("assets", {}).get("locations", [])}
    props_by_id = {p["id"]: p for p in project.get("assets", {}).get("props", [])}

    for s_idx, scene in enumerate(project.get("scenes", []), start=1):
        if scene.get("shots"):
            continue
        loc = locs_by_id.get(scene.get("location_id", ""), {})
        loc_desc = loc.get("description") or scene.get("plot_summary") or scene.get("name", "")
        dop = scene.get("dop_direction", {})
        lighting = dop.get("lighting_setup") or loc.get("lighting_default") or "cinematic motivated lighting"
        state_lookup = {cs["character_id"]: cs for cs in scene.get("character_states", [])}

        shots: list[dict[str, Any]] = []
        for b_idx, beat in enumerate(scene.get("beats", []), start=1):
            shot_id = f"s{s_idx:02d}_sh{b_idx:02d}"
            primary_cid = beat.get("character_id") or (scene.get("character_ids") or [None])[0]
            bound_cids = [primary_cid] if primary_cid else list(scene.get("character_ids") or [])
            ch = chars_by_id.get(primary_cid or "", {})
            cname = ch.get("name") or "Subject"
            c_state = state_lookup.get(primary_cid or "", {})
            wardrobe = c_state.get("resolved_wardrobe") or ch.get("default_wardrobe") or ch.get("appearance") or ""
            prop_ids = c_state.get("resolved_prop_ids") or scene.get("prop_ids") or []
            prop_names = [props_by_id[pid]["name"] for pid in prop_ids if pid in props_by_id]
            prop_clause = f" Active props in frame: {', '.join(prop_names)}." if prop_names else ""
            action = beat.get("blocking_action") or scene.get("plot_summary") or "stands in frame"
            spoken = beat.get("spoken_text")

            start_prompt = (
                f"{medium}. Medium cinematic shot on {cam_body} with 50mm {lens_pkg} at {aperture}. "
                f"{cname} ({wardrobe}) {action}.{prop_clause} "
                f"Environment: {loc_desc}. Lighting: {lighting}."
            ).strip()

            motion_prompt = (
                f"Smooth cinematic camera movement ({dop.get('camera_energy') or 'slow dolly push-in'}). "
                f"{cname} performs the beat naturally: {action}."
            ).strip()

            audio_cue = (
                f"{cname} ({beat.get('tone') or 'natural'}): '{spoken}'"
                if spoken
                else f"Ambient room tone and natural foley for {scene.get('name', 'scene')}."
            )

            shots.append(
                {
                    "id": shot_id,
                    "order": b_idx,
                    "beat_ids": [beat["id"]],
                    "duration_sec": 6,
                    "framing": "Medium cinematic shot",
                    "focal_length_mm": 50,
                    "camera_motion": {
                        "description": dop.get("camera_energy") or "Slow dolly push-in",
                        "higgsfield_motion_id": "slow_push_in",
                        "strength": 0.4,
                    },
                    "bindings": {
                        "character_ids": [c for c in bound_cids if c],
                        "higgsfield_character_ids": [],
                        "reference_image_urls": [],
                    },
                    "start_frame": {
                        "prompt": start_prompt,
                        "higgsfield_model": "soul_v2",
                        "request_id": None,
                        "image_url": None,
                        "local_path": None,
                        "approved": False,
                    },
                    "end_frame": {
                        "prompt": None,
                        "image_url": None,
                        "local_path": None,
                    },
                    "video": {
                        "higgsfield_model": "bytedance/seedance-2.5/image-to-video",
                        "motion_prompt": motion_prompt,
                        "audio_cue": audio_cue,
                        "generate_audio": True,
                        "quality": "1080p",
                        "seed": None,
                        "request_id": None,
                        "status": "pending",
                        "video_url": None,
                        "local_path": None,
                    },
                    "review": {
                        "approved": False,
                        "notes": None,
                    },
                }
            )
        scene["shots"] = shots


def stage_bible_into_comfy_project(
    bible: dict[str, Any],
    project_name: str,
    settings: FactorySettings,
    source_file_dir: Path | None = None,
) -> dict[str, Any]:
    """Persist project.json into ~/ComfyFactory/outputs/<project_name>/ and stage all shot manifests for Director Studio."""
    resolve_scene_continuity(bible)
    synthesize_missing_shots(bible)
    resolve_scene_continuity(bible)

    pdir = settings.outputs_dir / project_name
    pdir.mkdir(parents=True, exist_ok=True)
    chars_root = pdir / "characters"
    chars_root.mkdir(parents=True, exist_ok=True)

    # Check available local LoRAs in ~/ComfyFactory/loras/
    available_loras: dict[str, list[str]] = {}
    if settings.loras_dir.exists():
        for d in settings.loras_dir.iterdir():
            if d.is_dir():
                files = [f.name for f in sorted(d.glob("*.safetensors"))]
                if files:
                    available_loras[d.name.lower()] = files

    char_specs_by_id: dict[str, CharacterSpec] = {}
    for ch in bible.get("assets", {}).get("characters", []):
        cid = ch["id"]
        cname = ch["name"]
        cdir = chars_root / cid
        cdir.mkdir(parents=True, exist_ok=True)
        casting = ch.setdefault("casting", {})
        nw = casting.setdefault("nairobiwood", {})

        # Copy or link a master portrait if one exists in user_reference_photos and master_portrait_path isn't set yet
        master_local = cdir / "master_portrait.png"
        existing_mp = casting.get("master_portrait_path")
        if existing_mp and Path(existing_mp).exists():
            master_local = Path(existing_mp)
        elif not master_local.exists() and source_file_dir is not None:
            for ref_rel in casting.get("user_reference_photos") or []:
                if str(ref_rel).startswith(("http://", "https://")):
                    continue
                cand = (source_file_dir / str(ref_rel)).resolve()
                if cand.exists() and cand.is_file():
                    shutil.copy2(cand, master_local)
                    break

        if master_local.exists():
            casting["master_portrait_path"] = str(master_local)

        # Auto-match character name/slug against ~/ComfyFactory/loras/ if not explicitly set
        lora_candidate = nw.get("lora_name")
        if not lora_candidate:
            for key_try in (cname.lower(), slugify(cname), cid.removeprefix("char_")):
                if key_try in available_loras:
                    lora_candidate = key_try
                    break

        has_lora = bool(lora_candidate and lora_candidate.lower() in available_loras)
        if has_lora:
            mode = "lora"
            nw["lora_name"] = lora_candidate
        elif casting.get("master_portrait_path"):
            mode = "kontext_portrait"
        else:
            mode = nw.get("mode") or "kontext_portrait"
        nw["mode"] = mode

        char_specs_by_id[cid] = CharacterSpec(
            char_id=cid,
            name=cname,
            mode=mode,
            trigger_word=nw.get("trigger_word") or "",
            lora_strength=float(nw.get("lora_strength") or 0.85),
            appearance=ch.get("appearance") or "",
            default_wardrobe=ch.get("default_wardrobe") or ch.get("appearance") or "",
            default_prop_ids=list(ch.get("default_prop_ids") or []),
            master_portrait_path=casting.get("master_portrait_path"),
        )

    # Re-run continuity so newly discovered master_portrait_paths propagate to shot bindings
    resolve_scene_continuity(bible)

    bible_path = pdir / "project.json"
    bible_path.write_text(json.dumps(bible, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    props_by_id = {p["id"]: p for p in bible.get("assets", {}).get("props", [])}
    shot_order: list[str] = []
    staged_count = 0

    for scene in sorted(bible.get("scenes", []), key=lambda s: s.get("order", 0)):
        scene_id = scene["id"]
        scene_name = scene.get("name") or scene_id
        beats_by_id = {b["id"]: b for b in scene.get("beats", [])}
        state_by_cid = {cs["character_id"]: cs for cs in scene.get("character_states", [])}

        for s_order, shot in enumerate(sorted(scene.get("shots", []), key=lambda sh: sh.get("order", 0)), start=1):
            shot_id = shot["id"]
            shot_order.append(shot_id)
            sdir = pdir / "shots" / shot_id
            sdir.mkdir(parents=True, exist_ok=True)
            kf_dir = sdir / "keyframes"
            kf_dir.mkdir(parents=True, exist_ok=True)

            bindings = shot.get("bindings", {})
            bound_cids = bindings.get("character_ids") or scene.get("character_ids") or []
            primary_cid = bound_cids[0] if bound_cids else ""
            primary_spec = char_specs_by_id.get(primary_cid)
            char_name = (
                (bible_char_lora_or_name(primary_cid, bible, primary_spec))
                if primary_cid
                else "ensemble"
            )
            char_mode = primary_spec.mode if primary_spec else "description_only"

            c_state = state_by_cid.get(primary_cid, {})
            inh_wardrobe = c_state.get("resolved_wardrobe") or (primary_spec.default_wardrobe if primary_spec else "")
            inh_pids = c_state.get("resolved_prop_ids") or list(scene.get("prop_ids") or [])
            inh_prop_labels = [
                f"{props_by_id[pid]['name']}" if pid in props_by_id else pid
                for pid in inh_pids
            ]

            beat_texts: list[str] = []
            spoken_lines: list[str] = []
            for bid in shot.get("beat_ids") or []:
                b = beats_by_id.get(bid)
                if b:
                    if b.get("blocking_action"):
                        beat_texts.append(b["blocking_action"])
                    if b.get("spoken_text"):
                        spoken_lines.append(b["spoken_text"])

            sf = shot.get("start_frame") or {}
            ef = shot.get("end_frame") or {}
            vid = shot.get("video") or {}

            start_prompt_raw = sf.get("prompt") or ""
            # Ensure inherited wardrobe & persistent props are present in the prepared start_prompt
            continuity_notes: list[str] = []
            if inh_wardrobe and inh_wardrobe[:24].lower() not in start_prompt_raw.lower():
                continuity_notes.append(f"Wardrobe: {inh_wardrobe}")
            for plabel in inh_prop_labels:
                if plabel.lower() not in start_prompt_raw.lower():
                    continuity_notes.append(f"Prop: {plabel}")
            start_prompt = (
                f"{start_prompt_raw} [{' | '.join(continuity_notes)}]"
                if continuity_notes and start_prompt_raw
                else (start_prompt_raw or "; ".join(continuity_notes))
            )

            end_prompt = ef.get("prompt") or None
            motion_prompt = vid.get("motion_prompt") or shot.get("camera_motion", {}).get("description") or ""
            audio_cue = vid.get("audio_cue") or ""

            # Copy existing local keyframe if importing a StudioMode run that already has local images
            start_local = kf_dir / "start_frame.png"
            if not start_local.exists() and source_file_dir is not None and sf.get("local_path"):
                src_kf = (source_file_dir / str(sf["local_path"])).resolve()
                if src_kf.exists() and src_kf.is_file():
                    shutil.copy2(src_kf, start_local)

            m_path = sdir / "shot.manifest.json"
            if m_path.exists():
                existing = ShotManifest.model_validate_json(m_path.read_text(encoding="utf-8"))
            else:
                existing = None

            has_start_img = start_local.exists() or bool(existing and existing.keyframes.start_frame_path and Path(existing.keyframes.start_frame_path).exists())
            resolved_start_path = (
                str(start_local)
                if start_local.exists()
                else (existing.keyframes.start_frame_path if existing else None)
            )
            approved = bool(existing.keyframes.approved if existing else sf.get("approved", False))
            kf_status = (
                "approved"
                if approved
                else ("awaiting_approval" if has_start_img else "staged")
            )

            kf_state = KeyframeState(
                status=kf_status,
                start_frame_path=resolved_start_path,
                end_frame_path=existing.keyframes.end_frame_path if existing else None,
                has_end_frame=bool(end_prompt or (existing and existing.keyframes.has_end_frame)),
                approved=approved,
                start_prompt=start_prompt,
                end_prompt=end_prompt,
                reference_anchor_path=primary_spec.master_portrait_path if primary_spec else None,
                seed=existing.keyframes.seed if existing else (42 + s_order),
                lora_strength=primary_spec.lora_strength if primary_spec else 0.85,
            )

            manifest = ShotManifest(
                project_name=project_name,
                shot_id=shot_id,
                scene_id=scene_id,
                scene_name=scene_name,
                shot_order=s_order,
                character_name=char_name,
                character_mode=char_mode,
                characters=[char_specs_by_id[c] for c in bound_cids if c in char_specs_by_id],
                prompt=motion_prompt or start_prompt,
                video_motion_prompt=motion_prompt,
                beat_summary=" ".join(beat_texts) or scene.get("plot_summary", ""),
                spoken_text=" / ".join(spoken_lines) if spoken_lines else None,
                audio_cue=audio_cue,
                inherited_wardrobe=inh_wardrobe,
                inherited_props=inh_prop_labels,
                duration_sec=float(shot.get("duration_sec") or 5.0),
                fps=float(bible.get("project", {}).get("technical_specs", {}).get("fps") or 24.0),
                total_frames=existing.total_frames if existing else 0,
                keyframes=kf_state,
                master_video_path=existing.master_video_path if existing else None,
                preview_video_path=existing.preview_video_path if existing else None,
                contact_strip_path=existing.contact_strip_path if existing else None,
                last_frame_path=existing.last_frame_path if existing else None,
                checkpoints=existing.checkpoints if existing else {},
                updated_at=utc_now_iso(),
            )
            m_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
            staged_count += 1

    story_manifest = StoryManifest(
        project_name=project_name,
        scene_description=bible.get("project", {}).get("logline", ""),
        wardrobe_notes="Managed per-scene in project.json character_states",
        lighting_notes=bible.get("style_guide", {}).get("lighting_philosophy", ""),
        shot_order=shot_order,
        updated_at=utc_now_iso(),
    )
    (pdir / "story_manifest.json").write_text(story_manifest.model_dump_json(indent=2), encoding="utf-8")

    return {
        "project_name": project_name,
        "bible_path": str(bible_path),
        "scene_count": len(bible.get("scenes", [])),
        "staged_shot_count": staged_count,
        "shot_order": shot_order,
        "characters": [spec.model_dump() for spec in char_specs_by_id.values()],
    }


def bible_char_lora_or_name(cid: str, bible: dict[str, Any], spec: CharacterSpec | None) -> str:
    for ch in bible.get("assets", {}).get("characters", []):
        if ch["id"] == cid:
            nw = ch.get("casting", {}).get("nairobiwood", {})
            if nw.get("lora_name"):
                return str(nw["lora_name"])
            return str(ch.get("name") or cid)
    return spec.name if spec else cid
