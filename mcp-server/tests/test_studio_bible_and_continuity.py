"""Tests for StudioMode project.json ingestion, scene-to-scene wardrobe/prop persistence, and Master Portrait Kontext anchoring."""

from __future__ import annotations

from pathlib import Path

from comfy_factory_mcp.config import FactorySettings, ensure_data_dirs
from comfy_factory_mcp.service import FactoryService
from comfy_factory_mcp.shot_director import ShotDirector
from comfy_factory_mcp.studio_bible import parse_raw_story_into_bible, resolve_scene_continuity
from test_shot_and_branch_director import FakeComfyClient, FakeTunnel


def test_scene_to_scene_wardrobe_and_prop_persistence() -> None:
    raw_story = {
        "Story": {
            "Title": "Persistence Test",
            "Logline": "Testing wardrobe and prop carryover across scenes.",
            "Character": [
                {
                    "Id": 1,
                    "Name": "D.Leston",
                    "Appearance": "Hispanic man with wavy dark hair and salt-and-pepper beard",
                    "Wardrobe": "Matte black lab shirt",
                    "Props": ["prop_laptop"],
                }
            ],
            "Props": [
                {"id": "prop_laptop", "Name": "MacBook Laptop", "Description": "Open silver laptop"},
                {"id": "prop_aviators", "Name": "Aviator Sunglasses", "Description": "Gold-rimmed aviators"},
            ],
            "Scenes": [
                {
                    "Id": 1,
                    "Scene_Name": "Scene 1: The Lab",
                    "Scene_plot": "Typing at night.",
                    "Scenes_Characters": ["D.Leston"],
                    "Scene_background": "Dark room with monitors",
                    "Scene_dialogues": [{"Sequential": "0", "Character_speaking": "D.Leston", "Text": "Line 1"}],
                },
                {
                    "Id": 2,
                    "Scene_Name": "Scene 2: Still in Lab (No Wardrobe/Prop Change)",
                    "Scene_plot": "Looks into the screen.",
                    "Scenes_Characters": ["D.Leston"],
                    "Scene_background": "Dark room with monitors",
                    "Scene_dialogues": [{"Sequential": "0", "Character_speaking": "D.Leston", "Text": "Line 2"}],
                },
                {
                    "Id": 3,
                    "Scene_Name": "Scene 3: The Override (Wardrobe & Prop Change)",
                    "Scene_plot": "Steps outside.",
                    "Scenes_Characters": ["D.Leston"],
                    "Scene_background": "Courtyard",
                    "Character_states": [
                        {
                            "character_id": "char_d_leston",
                            "wardrobe": "Black shirt covered in white watching eyes",
                            "prop_ids": ["prop_aviators"],
                        }
                    ],
                    "Scene_dialogues": [{"Sequential": "0", "Character_speaking": "D.Leston", "Text": "Line 3"}],
                },
                {
                    "Id": 4,
                    "Scene_Name": "Scene 4: Epilogue (Void Override — Must Persist Scene 3 State)",
                    "Scene_plot": "Sings under the tree.",
                    "Scenes_Characters": ["D.Leston"],
                    "Scene_background": "Ancient tree",
                    "Scene_dialogues": [{"Sequential": "0", "Character_speaking": "D.Leston", "Text": "Line 4"}],
                },
            ],
        }
    }

    bible = parse_raw_story_into_bible(raw_story, project_slug="persistence-test")
    scenes = bible["scenes"]
    assert len(scenes) == 4

    # Scene 1 uses default wardrobe & props
    s1_state = scenes[0]["character_states"][0]
    assert s1_state["resolved_wardrobe"] == "Matte black lab shirt"
    assert s1_state["resolved_prop_ids"] == ["prop_laptop"]

    # Scene 2 had no change -> persists Scene 1 wardrobe & props
    s2_state = scenes[1]["character_states"][0]
    assert s2_state["continuity_mode"] == "inherit_prior_scene"
    assert s2_state["resolved_wardrobe"] == "Matte black lab shirt"
    assert s2_state["resolved_prop_ids"] == ["prop_laptop"]

    # Scene 3 overrides wardrobe & props
    s3_state = scenes[2]["character_states"][0]
    assert s3_state["continuity_mode"] == "override"
    assert s3_state["resolved_wardrobe"] == "Black shirt covered in white watching eyes"
    assert s3_state["resolved_prop_ids"] == ["prop_aviators"]

    # Scene 4 had no change -> persists Scene 3 overridden wardrobe & props!
    s4_state = scenes[3]["character_states"][0]
    assert s4_state["continuity_mode"] == "inherit_prior_scene"
    assert s4_state["resolved_wardrobe"] == "Black shirt covered in white watching eyes"
    assert s4_state["resolved_prop_ids"] == ["prop_aviators"]
    s4_shot = scenes[3]["shots"][0]
    assert s4_shot["bindings"]["continuity"]["inherited_wardrobe"]["char_d_leston"] == (
        "Black shirt covered in white watching eyes"
    )
    assert s4_shot["bindings"]["continuity"]["inherited_prop_ids"]["char_d_leston"] == ["prop_aviators"]


async def test_master_portrait_kontext_anchoring_and_edit_suggestion_rebuild(tmp_path: Path) -> None:
    workflows_dir = Path(__file__).resolve().parents[2] / "workflows"
    settings = FactorySettings(factory_data_dir=tmp_path / "factory")
    ensure_data_dirs(settings)
    svc = FactoryService(settings=settings)

    raw_story = {
        "Story": {
            "Title": "Non-LoRA Consistency Film",
            "Character": [
                {
                    "Id": 1,
                    "Name": "Leinad",
                    "Appearance": "Cornrow braids, groomed beard, cream linen suit",
                    "Wardrobe": "Cream linen safari suit",
                }
            ],
            "Scenes": [
                {
                    "Id": 1,
                    "Scene_Name": "Courtyard Arrival",
                    "Scene_plot": "Leinad walks across the sunlit courtyard.",
                    "Scenes_Characters": ["Leinad"],
                    "Scene_background": "Sunlit terracotta courtyard",
                    "Scene_dialogues": [
                        {"Sequential": "0", "Character_speaking": "Leinad", "Text": "I am the signal."}
                    ],
                }
            ],
        }
    }

    staged = await svc.import_or_init_story_bible(project_name="leinad-film", raw_story_json=raw_story)
    assert staged["staged_shot_count"] == 1
    state = await svc.get_studio_state(project_name="leinad-film")
    assert state["bible"] is not None
    assert len(state["storyboard"]) == 1
    assert state["storyboard"][0]["keyframes"]["status"] == "staged"
    assert state["storyboard"][0]["video_motion_prompt"] != ""

    tunnel = FakeTunnel(tmp_path)
    comfy = FakeComfyClient(workflows_dir)
    director = ShotDirector(settings, comfy, tunnel=tunnel)  # type: ignore[arg-type]

    # 1. Generate Master Character Portrait (without LoRA -> LoraLoader bypassed)
    mp_res = await director.generate_character_portrait(
        project_name="leinad-film",
        char_id="char_leinad",
    )
    assert Path(mp_res["master_portrait_path"]).exists()
    # Verify LoraLoader node '4' was bypassed since no LoRA exists for Leinad
    assert "4" not in comfy.queued[-1]

    # 2. Generate Scene First Frame -> records Master Portrait reference and bypasses missing LoRA
    scene_manifests = await director.generate_scene_keyframes(project_name="leinad-film", scene_id="scene_01")
    assert len(scene_manifests) == 1
    sh_manifest = scene_manifests[0]
    assert sh_manifest.keyframes.status == "awaiting_approval"
    assert sh_manifest.keyframes.reference_anchor_path == mp_res["master_portrait_path"]
    last_wf = comfy.queued[-1]
    assert "4" not in last_wf

    # 3. Rebuild First Frame with an Editing Suggestion (anchored to the current frame via LoadImage + VAEEncode)
    rebuilt = await director.regenerate_shot_keyframe(
        project_name="leinad-film",
        shot_id="s01_sh01",
        which="start",
        edit_suggestion="Keep composition and cream suit, add warm golden lens flare on sunglasses",
    )
    assert rebuilt.keyframes.last_edit_suggestion == (
        "Keep composition and cream suit, add warm golden lens flare on sunglasses"
    )
    assert "22" in comfy.queued[-1] and comfy.queued[-1]["22"]["class_type"] == "VAEEncode"

    # 4. Area-selected Rebuild using SAM2 target text
    area_rebuilt = await director.regenerate_shot_keyframe(
        project_name="leinad-film",
        shot_id="s01_sh01",
        which="start",
        edit_suggestion="silver aviator sunglasses reflecting palm trees",
        mask_target_text="sunglasses",
    )
    assert area_rebuilt.keyframes.status == "awaiting_approval"
    repair_wf = comfy.queued[-1]
    assert repair_wf["4"]["inputs"]["text_input"] == "sunglasses"
    assert "11" not in repair_wf  # LoraLoader bypassed cleanly for non-LoRA character

    # 5. Approve & Render Video using the decoupled video_motion_prompt
    rendered = await director.approve_and_render_video(
        project_name="leinad-film",
        shot_id="s01_sh01",
        prompt="Slow low-angle hero arc around Leinad as he speaks calmly",
        duration_sec=3.375,
    )
    assert rendered.keyframes.approved is True
    assert rendered.keyframes.status == "approved"
    assert rendered.video_motion_prompt == "Slow low-angle hero arc around Leinad as he speaks calmly"
    vid_wf = comfy.queued[-1]
    assert vid_wf["5"]["inputs"]["text"] == "Slow low-angle hero arc around Leinad as he speaks calmly"
