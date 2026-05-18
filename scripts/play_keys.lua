-- mpv key bindings for the MarkovSound queue player.
-- INS  → ./keep <current file>   (boost caption, keep playing)
-- DEL  → ./verwijder <current file>  (untrain + delete, skip to next)
--
-- Project root is passed via the MARKOVSOUND_ROOT environment variable.

local utils = require 'mp.utils'

local root = os.getenv("MARKOVSOUND_ROOT")
if not root or root == "" then
    mp.msg.error("MARKOVSOUND_ROOT not set; key bindings disabled")
    return
end

local function run(cmd, path)
    local res = mp.command_native({
        name = "subprocess",
        args = { cmd, path },
        playback_only = false,
        capture_stdout = true,
        capture_stderr = true,
    })
    return res
end

local function basename(p)
    return p and p:match("([^/]+)$") or ""
end

-- ./play exports the claimed Audio/playing path so keep/verwijder act on
-- the queue item even if mpv normalizes its own path representation.
local function current_track_path()
    local orig = os.getenv("MARKOVSOUND_ORIGINAL_PATH")
    if orig and orig ~= "" then return orig end
    return mp.get_property("path")
end

local function keep_track()
    local path = current_track_path()
    if not path then return end
    local cmd = utils.join_path(root, "keep")
    mp.osd_message("⤴ keep " .. basename(path), 2)
    local res = run(cmd, path)
    if res and res.status ~= 0 then
        mp.osd_message("keep failed: " .. (res.stderr or ""), 3)
    end
end

local function delete_track()
    local path = current_track_path()
    if not path then return end
    local cmd = utils.join_path(root, "verwijder")
    mp.osd_message("⤵ verwijder " .. basename(path), 2)
    local res = run(cmd, path)
    if res and res.status ~= 0 then
        mp.osd_message("verwijder failed: " .. (res.stderr or ""), 3)
    end
    -- Quit playback so the outer bash loop picks the next queued track.
    mp.command("quit")
end

mp.add_key_binding("INS", "track_keep", keep_track)
mp.add_key_binding("DEL", "track_delete", delete_track)
