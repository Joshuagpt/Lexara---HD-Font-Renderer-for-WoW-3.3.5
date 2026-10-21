#include <cmath>
#include "../Logger.h"
#include "MSDF.h"
#include "MSDFFont.h"
#include "MSDFCache.h"
#include "MSDFManager.h"
#include "MSDFShaders.h"
#include "Utils.h"
#include "Hooks.h"
#include <ranges>

namespace {
    uint32_t g_runtimeVBSize = 0;

    // 3D text (names above heads): glyph quads are padded by the SDF spread, so neighbouring quads (and the
    // shadow copy) overlap in the same plane. With depth writes on, which quad wins the depth test depends on
    // float rounding, i.e. on the camera angle -> glyphs vanish / flicker. So depth writes are turned off
    // for 3D MSDF text, right before each draw call (the game may re-set the state at any time).
    //
    // Hotkeys for A/B comparison (Ctrl+Shift held, so they do not collide with game key bindings):
    //   Ctrl+Shift+F9  = toggle: sample atlas at mip level 0 only
    //   Ctrl+Shift+F10 = toggle: depth-write-off for 3D text (default ON)
    bool s_diagNoMip = false;
    bool s_diagNoZWrite = true;
    bool s_text3dActive = false;
    bool s_diagZSaved = false;
    DWORD s_diagSavedZ = TRUE;

    int s_dumpStrings = 0;  // remaining 3D strings to dump (armed by Ctrl+Shift+F8)
    int s_dumpDraws = 0;    // remaining 3D font draws to dump

    // Dumps the vertices exactly as the client just wrote them for the GPU. Layout assumed (same as the 1.12
    // dump in the sibling project): 24 bytes per vertex = pos(3 floats), colour(uint32), u, v.
    void DumpWritten3D(CGxString* s, int destPtr, int vertIndex, int vertCount) {
        if (s_dumpStrings <= 0 || !destPtr || vertCount <= 0 || vertCount > 512) return;
        --s_dumpStrings;
        const uint8_t* d = reinterpret_cast<const uint8_t*>(destPtr);
        auto vf = [&](int i) { return reinterpret_cast<const float*>(d + static_cast<size_t>(i) * 24); };
        auto okf = [](float f) { return f == f && f > -1e9f && f < 1e9f; };

        int quads = vertCount / 4, badUV = 0, degenerate = 0, badPos = 0;
        for (int q = 0; q < quads; ++q) {
            const float* a = vf(q * 4 + 0);
            const float* c = vf(q * 4 + 3);
            if (!(okf(a[4]) && okf(a[5]) && okf(c[4]) && okf(c[5])) ||
                std::fabs(a[4]) > 1.0f || std::fabs(a[5]) > 1.0f || std::fabs(c[4]) > 1.0f || std::fabs(c[5]) > 1.0f) ++badUV;
            if (!(okf(a[0]) && okf(a[1]) && okf(a[2]) && okf(c[0]) && okf(c[1]) && okf(c[2]))) ++badPos;
            const float dx = c[0] - a[0], dy = c[1] - a[1], dz = c[2] - a[2];
            if (dx * dx + dy * dy + dz * dz < 1e-10f) ++degenerate;
        }
        Log("[DUMP] 3D string text='%s' flags=%08X vertIndex=%d vertCount=%d quads=%d  badUV=%d degenerate=%d badPos=%d",
            s->m_text ? s->m_text : "", s->m_flags, vertIndex, vertCount, quads, badUV, degenerate, badPos);
        for (int q = 0; q < quads && q < 14; ++q) {
            const float* a = vf(q * 4 + 0);
            const float* c = vf(q * 4 + 3);
            Log("[DUMP]    q%-2d v0=(%.3f %.3f %.3f) v3=(%.3f %.3f %.3f) uv0=(%.5f %.5f) uv3=(%.5f %.5f) col=%08X",
                q, a[0], a[1], a[2], c[0], c[1], c[2], a[4], a[5], c[4], c[5],
                *reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(a) + 12));
        }
    }

    void DiagPollKeys() {
        static bool d9 = false, d10 = false, d8 = false;
        const bool mods = (GetAsyncKeyState(VK_CONTROL) & 0x8000) && (GetAsyncKeyState(VK_SHIFT) & 0x8000);
        const bool k9 = mods && (GetAsyncKeyState(VK_F9) & 0x8000) != 0;
        const bool k10 = mods && (GetAsyncKeyState(VK_F10) & 0x8000) != 0;
        if (k9 && !d9) { s_diagNoMip = !s_diagNoMip; Log("[DIAG] F9: noMip = %d", s_diagNoMip ? 1 : 0); }
        if (k10 && !d10) { s_diagNoZWrite = !s_diagNoZWrite; Log("[DIAG] F10: noZWrite = %d", s_diagNoZWrite ? 1 : 0); }
        d9 = k9; d10 = k10;

        // Ctrl+Shift+F8: dump what is actually sent to the GPU for the 3D strings of the next frame(s).
        // Press it while a name is visibly missing characters, then compare it with a name that looks fine.
        const bool k8 = mods && (GetAsyncKeyState(VK_F8) & 0x8000) != 0;
        if (k8 && !d8) {
            s_dumpStrings = 40;
            s_dumpDraws = 8;
            Log("[DUMP] ---- F8 pressed: dumping up to %d 3D strings and %d draws ----", s_dumpStrings, s_dumpDraws);
        }
        d8 = k8;
    }

    // Rate-limited diagnostics. The old "first N lines" counters were used up on the login screen
    // (300 "waiting for glyphs" lines in 9 s), so nothing was left for in-game 3D name text.
    bool LogEvery(ULONGLONG& last, ULONGLONG intervalMs) {
        const ULONGLONG now = GetTickCount64();
        if (now - last < intervalMs) return false;
        last = now;
        return true;
    }
    uint32_t s_stat2dWaiting = 0, s_stat3dWaiting = 0, s_stat2dStamped = 0, s_stat3dStamped = 0;
    uint32_t s_statStale = 0, s_statStampedBad = 0;

    IDirect3DPixelShader9* s_cachedPS = nullptr;
    IDirect3DVertexShader9* s_cachedVS = nullptr;

    auto(*CGxString__CheckGeometry_call)() = reinterpret_cast<void(*)()>(0x006C4B09);
    constexpr uintptr_t CGxString__CheckGeometry_call_jmpback = 0x006C4B10;

    auto(*CGxString__CheckGeometry_site)() = reinterpret_cast<void(*)()>(0x006C4AF3);
    constexpr uintptr_t CGxString__CheckGeometry_site_loopstart = 0x006C4B00;

    auto(*CGxString__GetGlyphYMetrics_site)() = reinterpret_cast<void(*)()>(0x006C8C71);
    constexpr uintptr_t CGxString__GetGlyphYMetrics_site_jmpback = 0x006C8C77;

    auto(*CGxDevice__AllocateFontIndexBuffer_site)() = reinterpret_cast<void(*)()>(0x006C480C);
    constexpr uintptr_t CGxDevice__AllocateFontIndexBuffer_site_jmpback = 0x006C4811;

    auto(*CGxDevice__InitFontIndexBuffer_site)() = reinterpret_cast<void(*)()>(0x006C47BD);
    constexpr uintptr_t CGxDevice__InitFontIndexBuffer_site_jmpback = 0x006C47D8;

    auto(*IGxuFontProcessBatch_site)() = reinterpret_cast<void(*)()>(0x006C4CC4);
    constexpr uintptr_t IGxuFontProcessBatch_site_jmpback = 0x006C4CC9;

    auto(*CGxDevice__BufStream_site)() = reinterpret_cast<void(*)()>(0x006C4B40);
    constexpr uintptr_t CGxDevice__BufStream_site_jmpback = 0x006C4B45;

    auto(*bufalloc_1_site)() = reinterpret_cast<void(*)()>(0x006C4B64);
    constexpr uintptr_t bufalloc_1_site_jmpback = 0x006C4B70;

    auto(*bufalloc_2_site)() = reinterpret_cast<void(*)()>(0x006C4C67);
    constexpr uintptr_t bufalloc_2_site_jmpback = 0x006C4C8A;

    auto(*bufalloc_3_site)() = reinterpret_cast<void(*)()>(0x006C4C36);
    constexpr uintptr_t bufalloc_3_site_jmpback = 0x006C4C4B;

    bool s_msdfInitHookArmed = false;
    std::vector<uint32_t> s_prefetchPayload;

    // CGxString::m_flags high byte:
    //   bit 31 : geometry processed ("stamped")
    //   bit 30 : pending (0x40000000, set by InitializeTextLine)
    //   bits 24-29 : unused. Whether a stamped string is still valid is tracked by MSDFFont
    //                (per-string tile mask + eviction log), not by a version number in the flags.
    constexpr uint32_t kPendingFlag = 0x40000000;

    // 把 UTF-8 文本解码为 Unicode 码点（中文客户端文本为 UTF-8 多字节）
    void AppendUtf8Codepoints(const char* text, std::vector<uint32_t>& out) {
        const uint8_t* s = reinterpret_cast<const uint8_t*>(text);
        while (*s) {
            uint32_t cp = 0;
            int n = 0;
            if (*s < 0x80)               { cp = *s;        n = 1; }
            else if ((*s & 0xE0) == 0xC0) { cp = *s & 0x1F; n = 2; }
            else if ((*s & 0xF0) == 0xE0) { cp = *s & 0x0F; n = 3; }
            else if ((*s & 0xF8) == 0xF0) { cp = *s & 0x07; n = 4; }
            else { ++s; continue; }

            int i = 1;
            for (; i < n && (s[i] & 0xC0) == 0x80; ++i) {
                cp = (cp << 6) | (s[i] & 0x3F);
            }
            if (i < n) { s += i; continue; } // 非法或被截断的序列
            out.push_back(cp);
            s += n;
        }
    }

    // Called from the asm hook: run evictions deferred from the previous batch before validating strings
    void __cdecl FlushFontEvictions() {
        MSDFFont::FlushPendingEvictions();
        MSDFFont::StepDiskWarmup();
    }

    void __cdecl PrefetchCodepoints(CGxString* pThis) {
        if (s_prefetchPayload.empty()) return;
        if (!pThis || reinterpret_cast<uintptr_t>(pThis) & 1) return;

        if (MSDFFont* fontHandle = MSDFFont::Get(pThis->GetFontFace())) {
            std::ranges::sort(s_prefetchPayload);
            s_prefetchPayload.erase(std::ranges::unique(s_prefetchPayload).begin(), s_prefetchPayload.end());
            for (uint32_t codepoint : s_prefetchPayload) {
                fontHandle->GetGlyph(codepoint);
            }
        }
        s_prefetchPayload.clear();
    }

    void __fastcall ProcessGeometry(CGxString* pThis) {
        MSDFFont* fontHandle = MSDFFont::Get(pThis->GetFontFace());
        if (!fontHandle) return;

        // Already stamped and not pending -> nothing to do. Strings whose tiles were evicted are
        // detected and rebuilt by CheckGeometryHk; never re-stamp them here or stale UVs would be locked in.
        // Only "pending" strings, or strings with neither stamp nor pending bit (just rebuilt), are scanned.
        {
            const uint32_t hi = pThis->m_flags >> 24;
            if ((hi & 0x80) && !(hi & 0x40)) {
                // DIAG: a stamped string must contain no unprocessed quads (u > 1). If it does, the game rebuilt
                // the geometry without going through InitializeTextLine, and those glyphs will render empty.
                CGxFontGeomBatch* b = pThis->m_geomBuffers[0];
                if (b && b->m_verts.m_data && b->m_verts.m_count >= 4) {
                    uint32_t bad = 0;
                    for (uint32_t q = 0; q < b->m_verts.m_count; q += 4) {
                        if (b->m_verts.m_data[q].u > 1.0f) ++bad;
                    }
                    static int s_n = 0;
                    if (bad) ++s_statStampedBad;
                    if (bad && s_n < 100) {
                        ++s_n;
                        Log("[DIAG] STAMPED string has %u/%u unprocessed quads, is3d=%d, flags=%08X, text='%s'",
                            bad, b->m_verts.m_count / 4, (pThis->m_flags & 0x80) ? 1 : 0, pThis->m_flags,
                            pThis->m_text ? pThis->m_text : "");
                    }
                }
                return;
            }
        }
        fontHandle->BeginString(pThis);
        uint32_t tileMask = 0;

        CGxFontGeomBatch* batch = pThis->m_geomBuffers[0];
        if (!batch || batch->m_verts.m_count < 4) return;

        TSGrowableArray<CGxFontVertex>& verts = batch->m_verts;
        if (verts.m_count < 4) return;

        CGxFont* fontObj = pThis->m_fontObj;
        const uint32_t flags = fontObj->m_atlasPages[0].m_flags;
        const bool is3d = pThis->m_flags & 0x80;
        const double fontSizeMult = pThis->m_fontSizeMult;
        const double fontOffs = !is3d ? ((flags & 8) ? 4.5 : ((flags & 1) ? 2.5 : 0.0)) : 0.0;
        const double baselineOffs = (fontOffs > 0.0) ? 1.0 : 0.0;
        // 去掉 0.98 缩放，减少中文被压缩导致的裁切
        const double scale = (is3d ? fontSizeMult : CGxuFont::GetFontEffectiveHeight(is3d, fontSizeMult) * 1.0) / MSDF::SDF_RENDER_SIZE;
        const double pad = MSDF::SDF_SPREAD * scale;
        bool incomplete = false; // 有字形暂时拿不到 -> 不盖章，下一帧重试
        uint32_t missingCp = 0;  // DIAG: first codepoint that could not be resolved
        int missingWhy = 0;      // DIAG: 1 = no MSDF glyph, 2 = no game glyph entry

        for (uint32_t q = 0; q < verts.m_count; q += 4) {
            CGxFontVertex* vBase = &verts.m_data[q];
            if (vBase[0].u > 1.0f) {
                const uint32_t codepoint = vBase[0].u - 1.0f;

                // 3D nameplates generate missing glyphs on the render thread so mid-name characters
                // (e.g. 世界 in 旧世界武器军需官) do not stay blank for many frames waiting on workers.
                const GlyphMetrics* gm = fontHandle->GetGlyph(codepoint, /*preferSync=*/is3d);
                if (!gm) { incomplete = true; if (!missingCp) { missingCp = codepoint; missingWhy = 1; } continue; }

                CGxGlyphCacheEntry* entry = fontObj->GetOrCreateGlyphEntry(codepoint);
                if (!entry) { incomplete = true; if (!missingCp) { missingCp = codepoint; missingWhy = 2; } continue; }
                if (gm->atlasTile != 0xFF) tileMask |= 1u << gm->atlasTile;

                // Glyphs start as plain SDFs, which round sharp corners when drawn larger than the cell's render size.
                // 2D text drawn that large asks for the real MSDF; it replaces the SDF in the atlas in place.
                // (3D text is left out: its size on screen is not known here.)
                if (!is3d && scale > MSDF::MSDF_ABOVE_SCALE && !gm->msdf && gm->width > 0) {
                    fontHandle->RequestRefinement(codepoint);
                }

                CGxFontVertex* vert0 = &vBase[0];
                CGxFontVertex* vert1 = &vBase[1];
                CGxFontVertex* vert2 = &vBase[2];
                CGxFontVertex* vert3 = &vBase[3];

                const double leftOffs = fontObj->GetBearingX(entry, is3d, fontSizeMult);
                const double bitmapLeft = is3d ? leftOffs : gm->bitmapLeft * scale - leftOffs;

                // 使用 MSDF 的真实宽度，不再钳制左右边界：
                // 钳制会改变四边形宽度而 UV 不变，造成字形横向压缩/裁切。
                // 四边形外圈的 padding 在 SDF 中是透明的，重叠不影响显示。
                const double origLeft  = static_cast<double>(vert0->pos.X);
                const double msdfWidth = gm->width * scale;

                const double newLeft  = origLeft + (bitmapLeft != leftOffs ? bitmapLeft + 1.0 : 0.0) - pad + fontOffs * 0.5;
                const double newRight = newLeft + msdfWidth;

                const double newTop = static_cast<double>(vert1->pos.Y) + (gm->bitmapTop * scale) + pad - baselineOffs;
                const double newBottom = newTop - (gm->height * scale);

                vert0->pos.X = static_cast<float>(newLeft);  vert0->pos.Y = static_cast<float>(newBottom);
                vert1->pos.X = static_cast<float>(newLeft);  vert1->pos.Y = static_cast<float>(newTop);
                vert2->pos.X = static_cast<float>(newRight); vert2->pos.Y = static_cast<float>(newBottom);
                vert3->pos.X = static_cast<float>(newRight); vert3->pos.Y = static_cast<float>(newTop);

                const float u0 = gm->u0;
                const float u1 = gm->u1;
                const float v0 = gm->v0;
                const float v1 = gm->v1;

                const float uSign = (gm->atlasPageIndex & 1) ? -1.0f : 1.0f;
                const float vSign = (gm->atlasPageIndex & 2) ? -1.0f : 1.0f;

                vert0->u = u0 * uSign; vert0->v = v0 * vSign;
                vert1->u = u0 * uSign; vert1->v = v1 * vSign;
                vert2->u = u1 * uSign; vert2->v = v0 * vSign;
                vert3->u = u1 * uSign; vert3->v = v1 * vSign;
            }
        }

        // 有字形没拿到：保留现有标记、不盖版本章，下一帧 CheckGeometry 会再次进来重试
        // Remember which tiles this string uses (also for partial passes, so a later eviction is noticed).
        fontHandle->AccumulateString(pThis, tileMask, !incomplete);

        // Some glyphs unavailable: keep the pending bit and do not stamp; retried next frame.
        if (incomplete) {
            // Diagnostics for "missing characters": rate-limited per kind (3D and 2D separately), no fixed cap.
            if (is3d) ++s_stat3dWaiting; else ++s_stat2dWaiting;
            static ULONGLONG s_last2d = 0, s_last3d = 0;
            // 3D: 100ms so short-lived nameplate misses are more likely to appear in the log
            if (LogEvery(is3d ? s_last3d : s_last2d, is3d ? 100 : 3000)) {
                Log("[MSDF] string waiting for glyphs: is3d=%d quads=%u firstMissing=U+%04X why=%d preferSync=%d text='%s'",
                    is3d ? 1 : 0, verts.m_count / 4, missingCp, missingWhy, is3d ? 1 : 0,
                    pThis->m_text ? pThis->m_text : "");
            }
            return;
        }

        // Stamp = processed. Validity is tracked by MSDFFont (tile mask + eviction log), not by the token.
        if (is3d) {
            ++s_stat3dStamped;
            static ULONGLONG s_lastStamp3d = 0;
            if (LogEvery(s_lastStamp3d, 250)) {
                Log("[MSDF] 3D string stamped: quads=%u text='%s'", verts.m_count / 4, pThis->m_text ? pThis->m_text : "");
            }
        } else {
            ++s_stat2dStamped;
        }
        pThis->m_flags = (pThis->m_flags & 0x00FFFFFF) | (0x80u << 24);
    }

    bool __fastcall CGxString__CheckGeometryHk(CGxString* pThis) {
        if (MSDFFont* fontHandle = MSDFFont::Get(pThis->GetFontFace())) {
            const bool stamped = ((pThis->m_flags >> 24) & 0x80) != 0;
            // Stale = a tile this string uses has been evicted since it was built: rebuild it.
            // Strings that use only surviving tiles stay valid and are not rebuilt.
            if (fontHandle->CheckString(pThis, stamped) == MSDFFont::StringState::Stale) {
                ++s_statStale;
                static int s_staleLogged = 0;
                if (s_staleLogged < 100) {
                    ++s_staleLogged;
                    Log("[DIAG] string stale -> rebuild: is3d=%d stamped=%d text='%s' (#%d)",
                        (pThis->m_flags & 0x80) ? 1 : 0, stamped ? 1 : 0, pThis->m_text ? pThis->m_text : "", s_staleLogged);
                }
                pThis->ClearInstanceData();
                pThis->m_flags &= 0x00FFFFFF;
            }
        }
        CGxFontGeomBatch* batch = pThis->m_geomBuffers[0];
        if (!batch || !batch->m_verts.m_data) return pThis->CheckGeometry();

        g_runtimeVBSize += pThis->GetVertCountForPage(0);
        return pThis->CheckGeometry();
    }

    void __fastcall CGxString__WriteGeometryHk(CGxString* pThis, void* edx, int destPtr, int index, int vertIndex, int vertCount) {
        pThis->WriteGeometry(destPtr, index, vertIndex, vertCount);

        MSDFFont* fontHandle = MSDFFont::Get(pThis->GetFontFace());
        if (!fontHandle) {
            if (IDirect3DDevice9* device = D3D::GetDevice()) {
                constexpr float resetControl[4] = { 0, 0, 0, 0 };
                device->SetPixelShaderConstantF(MSDF::SDF_SAMPLER_SLOT, resetControl, 1);
                device->SetVertexShaderConstantF(MSDF::SDF_SAMPLER_SLOT, resetControl, 1);
            }
            return;
        }

        IDirect3DDevice9* device = D3D::GetDevice();
        if (!device) return;

        for (uint32_t pageIdx = 0; pageIdx < fontHandle->GetAtlasPageCount(); ++pageIdx) {
            auto* atlasTexture = fontHandle->GetAtlasPage(pageIdx);
            if (atlasTexture && atlasTexture->texture) {
                uint32_t slot = (15 - MSDF::MAX_ATLAS_PAGES + 1) + pageIdx;
                device->SetTexture(slot, atlasTexture->texture);
                device->SetSamplerState(slot, D3DSAMP_ADDRESSU, D3DTADDRESS_CLAMP);
                device->SetSamplerState(slot, D3DSAMP_ADDRESSV, D3DTADDRESS_CLAMP);
                device->SetSamplerState(slot, D3DSAMP_MINFILTER, D3DTEXF_LINEAR);
                device->SetSamplerState(slot, D3DSAMP_MAGFILTER, D3DTEXF_LINEAR);
                // Auto-generated mipmaps report GetLevelCount() == 1, so use the font's flag instead.
                // Trilinear when mips exist (the shader picks the LOD via tex2Dlod); pages without mips just use level 0.
                device->SetSamplerState(slot, D3DSAMP_MIPFILTER, (MSDFFont::IsAutoMipEnabled() && !s_diagNoMip) ? D3DTEXF_LINEAR : D3DTEXF_NONE);
                device->SetSamplerState(slot, D3DSAMP_MIPMAPLODBIAS, 0);
                device->SetSamplerState(slot, D3DSAMP_MAXMIPLEVEL, 0);
            }
        }

        const uint32_t flags = pThis->m_fontObj->m_atlasPages[0].m_flags;
        const bool is3d = pThis->m_flags & 0x80;
        const float controlFlag[4] = {
            is3d ? pThis->m_fontObj->m_rasterTargetSize : static_cast<float>(CGxuFont::GetFontEffectiveHeight(is3d, pThis->m_fontSizeMult)),
            is3d ? 0.0f : ((flags & 8) ? 2.0f : ((flags & 1) ? 1.0f : 0.0f)),
            MSDF::SDF_SPREAD, MSDF::ATLAS_SIZE
        };
        device->SetPixelShaderConstantF(MSDF::SDF_SAMPLER_SLOT, controlFlag, 1);
        device->SetVertexShaderConstantF(MSDF::SDF_SAMPLER_SLOT, controlFlag, 1);

        if (is3d) s_text3dActive = true; // OR over the whole batch; reset in CGxuFontRenderBatchHk
        if (is3d && s_dumpStrings > 0) {
            Log("[DUMP] control written for this string: (%.2f %.2f %.2f %.0f)  fontSizeMult=%.3f rasterTarget=%.2f",
                controlFlag[0], controlFlag[1], controlFlag[2], controlFlag[3],
                pThis->m_fontSizeMult, pThis->m_fontObj->m_rasterTargetSize);
            DumpWritten3D(pThis, destPtr, vertIndex, vertCount);
        }
    }

    void __fastcall CGxuFontRenderBatchHk(CGxuFont* pThis) {
        DiagPollKeys();
        {
            static ULONGLONG s_lastStat = 0;
            if (LogEvery(s_lastStat, 2000) && (s_stat2dWaiting | s_stat3dWaiting | s_stat2dStamped | s_stat3dStamped | s_statStale | s_statStampedBad)) {
                Log("[STAT] 2s: waiting 2d=%u 3d=%u | stamped 2d=%u 3d=%u | stale-rebuild=%u | stamped-with-unprocessed=%u",
                    s_stat2dWaiting, s_stat3dWaiting, s_stat2dStamped, s_stat3dStamped, s_statStale, s_statStampedBad);
                s_stat2dWaiting = s_stat3dWaiting = s_stat2dStamped = s_stat3dStamped = s_statStale = s_statStampedBad = 0;
            }
        }
        pThis->RenderBatch();
        s_text3dActive = false;
        if (s_diagZSaved) {
            if (IDirect3DDevice9* d = D3D::GetDevice()) d->SetRenderState(D3DRS_ZWRITEENABLE, s_diagSavedZ);
            s_diagZSaved = false;
        }
        if (IDirect3DDevice9* device = D3D::GetDevice()) {
            constexpr float resetControl[4] = { 0, 0, 0, 0 };
            device->SetPixelShaderConstantF(MSDF::SDF_SAMPLER_SLOT, resetControl, 1);
            device->SetVertexShaderConstantF(MSDF::SDF_SAMPLER_SLOT, resetControl, 1);
        }
    }

    char __cdecl GxuFontGlyphRenderGlyphHk(FT_Face fontFace, uint32_t fontSize, uint32_t codepoint, uint32_t pageInfo, CGxGlyphMetrics* resultBuffer, uint32_t outline_flag, uint32_t pad) {
        const char result = CGxuFont::RenderGlyph(fontFace, fontSize, codepoint, pageInfo, resultBuffer, outline_flag, pad);
        if (resultBuffer && MSDFFont::Get(fontFace)) {
            resultBuffer->m_bearingY -= resultBuffer->m_verAdv;
        }
        return result;
    }

    CGxGlyphCacheEntry* __fastcall CGxString__GetOrCreateGlyphEntryHk(CGxFont* fontObj, void* edx, uint32_t codepoint) {
        if (!fontObj) {
            return nullptr;
        }

        CGxGlyphCacheEntry* result = fontObj->GetOrCreateGlyphEntry(codepoint);
        if (result && MSDFFont::Get(CGxString::GetFontFace(fontObj->m_ftWrapper))) {
            result->m_metrics.u0 = 1.0f + codepoint;
            result->m_cellIndexMin = 0;
            result->m_cellIndexMax = 0;
            result->m_texturePageIndex = 0;
        }
        return result;
    }

    int __fastcall CGxString__InitializeTextLineHk(CGxString* pThis, void* edx, char* text, int textLength, int* a4, C3Vector* startPos, void* a6, int a7) {
        const int result = pThis->InitializeTextLine(text, textLength, a4, startPos, a6, a7);

        if (pThis->m_flags & kPendingFlag) return result;
        if (pThis->m_text) {
            AppendUtf8Codepoints(pThis->m_text, s_prefetchPayload);
        }
        pThis->m_flags |= kPendingFlag;
        return result;
    }

    __declspec(naked) void CGxString__CheckGeometry_siteHk() {
        __asm {
            pushad;
            call FlushFontEvictions;
            mov edi, ebx;

            test ebx, ebx;
            jnz pre_pass_loop;

            xor ebx, ebx;
            lea esp, [esp + 0];

        pre_pass_loop:
            test edi, edi;
            jz pre_pass_done;
            test di, 1;
            jnz pre_pass_done;

            mov ecx, edi;
            call CGxString__CheckGeometryHk;

            mov eax, [esi + 1Ch];
            add eax, edi;
            mov edi, [eax + 4];
            jmp pre_pass_loop;

        pre_pass_done:
            popad;
            push ebx;
            call PrefetchCodepoints;
            add esp, 4;
            jmp CGxString__CheckGeometry_site_loopstart;
        }
    }

    __declspec(naked) void CGxString__CheckGeometry_callHk() {
        __asm {
            mov ecx, ebx;
            call ProcessGeometry;
            jmp CGxString__CheckGeometry_call_jmpback;
        }
    }

    bool __cdecl MSDFFont_Get(FT_Face face) { return MSDFFont::Get(face); }
    __declspec(naked) void CGxString_GetGlyphYMetrics_siteHk() {
        __asm {
            mov edx, [ecx + 54h];
            pushad;
            push ecx;
            call MSDFFont_Get;
            add esp, 4;
            test al, al;
            popad;
            jz font_unsafe;
            xor ecx, ecx;
            jmp CGxString__GetGlyphYMetrics_site_jmpback;
        font_unsafe:
            mov ecx, [edx + 68h];
            jmp CGxString__GetGlyphYMetrics_site_jmpback;
        }
    }

    __declspec(naked) void CGxDevice__AllocateFontIndexBuffer_siteHk() {
        __asm {
            mov ebx, 3FFFh;
            jmp CGxDevice__AllocateFontIndexBuffer_site_jmpback;
        }
    }
    __declspec(naked) void CGxDevice__InitFontIndexBuffer_siteHk() {
        __asm {
            push 30000h;
            push 0;
            push 1;
            call CGxDevice::PoolCreateFn;
            mov ecx, dword ptr ds : [0C5DF88h]
            push 0;
            push 1801Ah;
            jmp CGxDevice__InitFontIndexBuffer_site_jmpback;
        }
    }
    __declspec(naked) void IGxuFontProcessBatch_siteHk() {
        __asm {
            mov g_runtimeVBSize, 0;
            pop ebx;
            pop esi;
            mov esp, ebp;
            pop ebp;
            jmp IGxuFontProcessBatch_site_jmpback;
        }
    }
    __declspec(naked) void CGxDevice__BufStream_siteHk() {
        __asm {
            mov eax, g_runtimeVBSize;
            cmp eax, 800h;
            jge check_upper;
            mov eax, 800h;
            jmp do_push;
        check_upper:
            cmp eax, 0FFFCh;
            jle do_push;
            mov eax, 0FFFCh;
        do_push:
            mov g_runtimeVBSize, eax;
            push eax;
            jmp CGxDevice__BufStream_site_jmpback;
        }
    }
    __declspec(naked) void bufalloc_1_siteHk() {
        __asm {
            xor eax, eax;
            mov esi, 0B4h;
            mov ebx, g_runtimeVBSize;
            jmp bufalloc_1_site_jmpback;
        }
    }
    __declspec(naked) void bufalloc_2_siteHk() {
        __asm {
            cmp ebx, g_runtimeVBSize;
            jz orig_skip;
            mov ecx, g_runtimeVBSize;
            sub ecx, ebx;
            push ecx;
            lea edx, [ebp - 18h];
            push edx;
            call CGxDevice::FlushBufferFn;
            add esp, 8;
            mov edi, eax;
            mov ebx, g_runtimeVBSize;
            jmp bufalloc_2_site_jmpback;
        orig_skip:
            jmp bufalloc_2_site_jmpback;
        }
    }
    __declspec(naked) void bufalloc_3_siteHk() {
        __asm {
            push g_runtimeVBSize;
            push eax;
            call CGxDevice::FlushBufferFn;
            add esp, 8;
            mov edi, eax;
            mov ebx, g_runtimeVBSize;
            jmp bufalloc_3_site_jmpback;
        }
    }

    int __cdecl FreeType_NewMemoryFaceHk(FT_Library library, const FT_Byte* file_base,
        FT_Long file_size, FT_Long face_index, FT_Face* aface) {
        if (!MSDF::g_realFtLibrary && FT_Init_FreeType(&MSDF::g_realFtLibrary) != 0) return -1;

        const int result = FT_New_Memory_Face(library, file_base, file_size, face_index, aface);
        if (result != 0 || !aface || !*aface) return result;

        MSDFFont::Register(*aface, file_base, file_size);
        return result;
    }

    int __cdecl FreeType_SetPixelSizesHk(FT_Face face, FT_UInt pixel_width, FT_UInt pixel_height) {
        return FT_Set_Pixel_Sizes(face, pixel_width, pixel_height);
    }

    int __cdecl FreeType_LoadGlyphHk(FT_Face face, FT_ULong glyph_index, FT_Int32 load_flags) {
        return FT_Load_Glyph(face, glyph_index, FT_LOAD_RENDER | FT_LOAD_NO_HINTING);
    }

    FT_UInt __cdecl FreeType_GetCharIndexHk(FT_Face face, FT_ULong charcode) {
        return FT_Get_Char_Index(face, charcode);
    }

    int __cdecl FreeType_GetKerningHk(FT_Face face, FT_UInt left_glyph, FT_UInt right_glyph, FT_UInt kern_mode, FT_Vector* akerning) {
        return FT_Get_Kerning(face, left_glyph, right_glyph, kern_mode, akerning);
    }

    int __cdecl FreeType_Done_FaceHk(FT_Face face) {
        MSDFFont::Unregister(face);
        return FT_Done_Face(face);
    }

    int __cdecl FreeType_Done_FreeTypeHk(FT_Library library) {
        MSDFFont::Shutdown();
        if (MSDF::g_msdfFreetype) {
            msdfgen::deinitializeFreetype(MSDF::g_msdfFreetype);
            MSDF::g_msdfFreetype = nullptr;
        }
        if (MSDF::g_realFtLibrary) {
            FT_Done_FreeType(MSDF::g_realFtLibrary);
            MSDF::g_realFtLibrary = nullptr;
        }
        return 0;
    }

    int __cdecl FreeType_NewFaceHk(int* library, int face_descriptor_ptr) {
        return 1;
    }

    // 3D text depth fix. MUST be registered from MSDF::initialize() (i.e. at DLL attach, before the
    // device exists): D3D.cpp only detours DrawIndexedPrimitive in CGxDevice::DeviceCreate if somebody
    // is already listening at that moment. Registering from FreeType_InitHk can be too late, and then
    // this callback never runs (the Ctrl+Shift+F10 toggle would change nothing).
    void RegisterText3DDepthHook() {
        static bool s_done = false;
        if (s_done) return;
        s_done = true;
        D3D::RegisterDrawIndexedPrimitiveCallback([](IDirect3DDevice9* dev, D3DPRIMITIVETYPE, INT, UINT, UINT, UINT, UINT) {
            if (!s_text3dActive) return;
            if (s_dumpDraws > 0) {
                --s_dumpDraws;
                float ctl[4] = {};
                dev->GetPixelShaderConstantF(MSDF::SDF_SAMPLER_SLOT, ctl, 1);
                DWORD zen = 0, zw = 0, cull = 0, blend = 0, atest = 0, zfunc = 0;
                dev->GetRenderState(D3DRS_ZENABLE, &zen); dev->GetRenderState(D3DRS_ZWRITEENABLE, &zw);
                dev->GetRenderState(D3DRS_ZFUNC, &zfunc); dev->GetRenderState(D3DRS_CULLMODE, &cull);
                dev->GetRenderState(D3DRS_ALPHABLENDENABLE, &blend); dev->GetRenderState(D3DRS_ALPHATESTENABLE, &atest);
                IDirect3DBaseTexture9* t[4] = {};
                for (int i = 0; i < 4; ++i) dev->GetTexture((15 - MSDF::MAX_ATLAS_PAGES + 1) + i, &t[i]);
                Log("[DUMP] 3D font draw: ctl_ps=(%.2f %.2f %.2f %.0f) ZENABLE=%lu ZFUNC=%lu ZWRITE(before our override)=%lu CULL=%lu BLEND=%lu ATEST=%lu tex[%p %p %p %p]",
                    ctl[0], ctl[1], ctl[2], ctl[3], zen, zfunc, zw, cull, blend, atest, t[0], t[1], t[2], t[3]);
                for (int i = 0; i < 4; ++i) if (t[i]) t[i]->Release();
            }
            static bool s_loggedFirst = false;
            if (!s_loggedFirst) { s_loggedFirst = true; Log("[DIAG] DIP hook: first 3D text draw seen (noZWrite=%d)", s_diagNoZWrite ? 1 : 0); }
            if (!s_diagNoZWrite) return;
            if (!s_diagZSaved) {
                dev->GetRenderState(D3DRS_ZWRITEENABLE, &s_diagSavedZ);
                s_diagZSaved = true;
            }
            dev->SetRenderState(D3DRS_ZWRITEENABLE, FALSE);
        });
    }

    int __cdecl FreeType_InitHk(void* memory, FT_Library* alibrary) {
        if (!MSDF::INITIALIZED) {
            std::string localeStr = MSDF::GetGameLocale();
            const char* locale = localeStr.c_str();
            MSDF::IS_CJK = locale && (strcmp(locale, "zhCN") == 0 ||
                strcmp(locale, "zhTW") == 0 ||
                strcmp(locale, "koKR") == 0);

            MSDF::INITIALIZED = true;

            // Previously CJK locales early-returned and disabled MSDF entirely.
            // Now we enable the full MSDF path for zhCN/zhTW/koKR as well.
            // Chinese/Korean still benefit from pre-generation (see MSDFPregen)
            // because the CJK range is large; without it the first appearance
            // of many glyphs can stutter.

            DetourTransactionBegin();
            DetourUpdateThread(GetCurrentThread());
            Hooks::Detour(&FreeType::NewMemoryFaceFn, FreeType_NewMemoryFaceHk);
            Hooks::Detour(&FreeType::NewFaceFn, FreeType_NewFaceHk);
            Hooks::Detour(&FreeType::Done_FaceFn, FreeType_Done_FaceHk);
            Hooks::Detour(&FreeType::SetPixelSizesFn, FreeType_SetPixelSizesHk);
            Hooks::Detour(&FreeType::GetCharIndexFn, FreeType_GetCharIndexHk);
            Hooks::Detour(&FreeType::LoadGlyphFn, FreeType_LoadGlyphHk);
            Hooks::Detour(&FreeType::GetKerningFn, FreeType_GetKerningHk);
            Hooks::Detour(&FreeType::Done_FreeTypeFn, FreeType_Done_FreeTypeHk);

            Hooks::Detour(&CGxuFont::RenderBatchFn, CGxuFontRenderBatchHk);
            Hooks::Detour(&CGxuFont::RenderGlyphFn, GxuFontGlyphRenderGlyphHk);
            Hooks::Detour(&CGxFont::GetOrCreateGlyphEntryFn, CGxString__GetOrCreateGlyphEntryHk);

            Hooks::Detour(&CGxDevice__AllocateFontIndexBuffer_site, CGxDevice__AllocateFontIndexBuffer_siteHk);
            Hooks::Detour(&CGxDevice__InitFontIndexBuffer_site, CGxDevice__InitFontIndexBuffer_siteHk);

            Hooks::Detour(&CGxString__GetGlyphYMetrics_site, CGxString_GetGlyphYMetrics_siteHk);
            Hooks::Detour(&CGxString__CheckGeometry_site, CGxString__CheckGeometry_siteHk);
            Hooks::Detour(&CGxString__CheckGeometry_call, CGxString__CheckGeometry_callHk);

            Hooks::Detour(&IGxuFontProcessBatch_site, IGxuFontProcessBatch_siteHk);
            Hooks::Detour(&CGxDevice__BufStream_site, CGxDevice__BufStream_siteHk);
            Hooks::Detour(&bufalloc_1_site, bufalloc_1_siteHk);
            Hooks::Detour(&bufalloc_2_site, bufalloc_2_siteHk);
            Hooks::Detour(&bufalloc_3_site, bufalloc_3_siteHk);

            Hooks::Detour(&CGxString::CheckGeometryFn, CGxString__CheckGeometryHk);
            Hooks::Detour(&CGxString::WriteGeometryFn, CGxString__WriteGeometryHk);
            Hooks::Detour(&CGxString::InitializeTextLineFn, CGxString__InitializeTextLineHk);
            DetourTransactionCommit();

            D3D::RegisterOnDestroy([]() {
                if (s_cachedPS) { s_cachedPS->Release(); s_cachedPS = nullptr; }
                if (s_cachedVS) { s_cachedVS->Release(); s_cachedVS = nullptr; }
                MSDFFont::ClearAllCache();
            });

            D3D::RegisterPixelShaderInit([](CGxDevice::ShaderData* shaderData) {
                if (!shaderData || shaderData != MSDF::g_FontPixelShader) return;
                if (!s_cachedPS) {
                    s_cachedPS = D3D::CompilePixelShader({
                        .shaderCode = pixelShaderHLSL,
                        .target = "ps_3_0"
                    });
                }
                if (s_cachedPS) {
                    if (shaderData->pixel_shader) shaderData->pixel_shader->Release();
                    s_cachedPS->AddRef();
                    shaderData->pixel_shader = s_cachedPS;
                    shaderData->compilation_flags = 1;
                }
            });

            s_cachedPS = D3D::CompilePixelShader({
                .shaderCode = pixelShaderHLSL,
                .target = "ps_3_0"
            });
            if (s_cachedPS && MSDF::g_FontPixelShader) {
                if (MSDF::g_FontPixelShader->pixel_shader) MSDF::g_FontPixelShader->pixel_shader->Release();
                s_cachedPS->AddRef();
                MSDF::g_FontPixelShader->pixel_shader = s_cachedPS;
                MSDF::g_FontPixelShader->compilation_flags = 1;
            }

            D3D::RegisterVertexShaderInit([](CGxDevice::ShaderData* shaderData) {
                if (!shaderData || shaderData != MSDF::g_FontVertexShader) return;
                if (!s_cachedVS) {
                    s_cachedVS = D3D::CompileVertexShader({
                        .shaderCode = vertexShaderHLSL,
                        .target = "vs_3_0"
                    });
                }
                if (s_cachedVS) {
                    if (shaderData->vertex_shader) shaderData->vertex_shader->Release();
                    s_cachedVS->AddRef();
                    shaderData->vertex_shader = s_cachedVS;
                    shaderData->compilation_flags = 1;
                }
            });

            s_cachedVS = D3D::CompileVertexShader({
                .shaderCode = vertexShaderHLSL,
                .target = "vs_3_0"
            });
            if (s_cachedVS && MSDF::g_FontVertexShader) {
                if (MSDF::g_FontVertexShader->vertex_shader) MSDF::g_FontVertexShader->vertex_shader->Release();
                s_cachedVS->AddRef();
                MSDF::g_FontVertexShader->vertex_shader = s_cachedVS;
                MSDF::g_FontVertexShader->compilation_flags = 1;
            }

            s_prefetchPayload.reserve(16383);
            CGxDevice::InitFontIndexBufferFn();
        }

        if (const FT_Error error = FT_Init_FreeType(&MSDF::g_realFtLibrary)) return error;

        if (alibrary) *alibrary = MSDF::g_realFtLibrary;

        MSDF::g_msdfFreetype = msdfgen::initializeFreetype();
        if (!MSDF::g_msdfFreetype) {
            FT_Done_FreeType(MSDF::g_realFtLibrary);
            MSDF::g_realFtLibrary = nullptr;
            return -1;
        }
        return 0;
    }
}

void MSDF::initialize() {
    if (s_msdfInitHookArmed) {
        return;
    }

    RegisterText3DDepthHook();
    Hooks::Detour(&FreeType::InitFn, FreeType_InitHk);
    s_msdfInitHookArmed = true;
}
